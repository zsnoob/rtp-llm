# DSV4 Mega CSA/HCA TP1 接入状态与后续方案

更新日期：2026-08-25

## 1. 当前结论

DSV4 Mega CSA 的开源框架适配已经进入生产 decode 层循环。TP1 单层真实 RTP attention
sublayer 的数值对照、eager、CUDA Graph 和 slot reuse 已通过。2026-08-18 起本地整模型
serving 端到端已跑通（裁层 DSV4-Pro）；2026-08-19 起 **全量 DeepSeek-V4-Flash（43 层，
单卡）端到端跑通**，baseline 与 Mega（CSA+HCA 双开）输出语义等价，仅在近平局 token 处
出现 greedy 分岔（与框架 smoke 对不同拓扑使用各自 golden 的既有现象同类）。框架公共
CUDA13 `rtp-kernel` wheel 仍不含 Mega 扩展；当前所有验证使用本地 wheel + 未提交的本地
lock 补丁，发布制品后才能推平公共依赖。

2026-08-19 起，同一 extension 二进制内同时编译 **DSV4-Pro 与 DSV4-Flash 两套 CSA 几何**
（Pro: dim 7168 / q_lora 1536 / 128 heads / o_groups 16；Flash: 4096 / 1024 / 64 / 8），
python wrapper 按张量形状 dispatch；HCA 算子自始就带双几何。RTP 侧 weights/adapter/runtime
以 `CSAGeometry` profile 按 `attn.dim` 选择，两种模型共用全部代码路径。

2026-08-18 起，HCA（`compress_ratio == 128`）层的同型接入也已完成并通过单层对照，由独立
开关 `DSV4_MEGA_HCA` 控制。HCA 没有 indexer/TopK/MQA 阶段：opA/opB 两个融合 GEMM 覆盖
front 投影、state 环 FRONT-EMIT、mHC post/comb tail、q_b 投影、128-token 边界压缩写
HCA_KV 和 window 写 SWA_KV；query RMSNorm+RoPE 使用 extension CUDA 算子
`q_rmsnorm_rope_cuda_`，稠密 compressed index 直接复用每 step 已构建的
`topk_total_by_ratio[128]`。CSA+HCA 同开时
Mega 覆盖 DSV4-Pro 全部 61 个 attention 层（30 CSA + 31 HCA）。

当前实现遵循“完整 attention sublayer 单独选路”，没有逐个替换普通算子：

```text
Block.forward_decode
  ├─ CSA Mega: adapter 已挂载 && q_len == 1
  │    mHC pre + attention RMSNorm
  │    -> front mixed GEMM
  │    -> WQ-B + indexer compressor + SWA write
  │    -> FP8 MQA + main compressor + query RMS/RoPE
  │    -> RTP persistent TopK
  │    -> RTP 原生 FlashMLA 路径
  │    -> CUDA inverse-RoPE + FP8 quant
  │    -> 现有 wo_a / wo_b output projection
  │    -> mHC post
  ├─ HCA Mega: adapter 已挂载 && q_len == 1
  │    mHC pre + attention RMSNorm
  │    -> front mixed GEMM（FRONT-EMIT 写 kv|gate state 环 + mHC post/comb tail）
  │    -> WQ-B + 边界 compressor 写 HCA_KV + window 写 SWA_KV
  │    -> CUDA q_rmsnorm_rope_cuda_（q RMSNorm + 部分 RoPE）
  │    -> metadata 稠密 compressed index（topk_total_by_ratio[128]）
  │    -> RTP 原生 FlashMLA 路径（SWA_KV + HCA_KV）
  │    -> CUDA inverse-RoPE + FP8 quant
  │    -> 现有 wo_a / wo_b output projection
  │    -> mHC post
  └─ 原路径: 其他所有情况
       attn_hc.pre -> AttentionFP8.forward_decode
       -> output projection -> attn_hc.post
```

当前 attention adapter 仍只替换 attention sublayer。本分支同时增加了可独立启用的
四-kernel native MoE front：在 DSV4-Pro decode、TP1、MegaMoE-SE 且底层 storage 满足
capacity-128 时，整体替换
`ffn_hc.pre -> ffn_norm -> router F.linear -> Triton gate-pack`，并直接发布到官方
DeepGEMM expert core 已有的 symmetric buffer。开关、回退条件和待完成的发布验证见下文
“B300 四-kernel MoE front 生产接入”。model head mHC 仍不在 Mega 替换范围内。

### 2026-08-26：CUDA13 wheel release gate

RTP 集成代码与 release metadata 已推送到 `zsnoob/rtp-llm:dsv4-mega`，当前提交为
`cbd82fd5d`。CUDA extension 的可发布源码固定为 `dsv4_megakernel@c81d23d`；本地已准备
可复现源码归档（SHA256
`e07f90609eedab9247983b7426726c20274edc1a7493f2d828122802a6046873`），其 MoE-front
source fingerprint 为
`5d9d222c7cd32c0969b2ed6cca25ffd837c07eddc07e3d204221179e32221ee0`。构建脚本只用
`torch.cuda.get_device_capability()` 检查四张设备均为 `(10,3)`，并在安装后验证完整
`rtp_ops`、`rtp_ops_dsv4_mega`、四-kernel contract 和 build identity。

本机已完成源码 fingerprint、脚本语法、Python compileall、文档 diff 和历史四-kernel
artifact 校验；真实 CUDA13/SM103 wheel 构建、EP4 MoE block 正确性以及 CSA+HCA+MoE
front 完整生成仍是 release gate，必须在可访问的四卡 WebIDE 上执行。当前 Mac 缺少
CUDA toolchain，现有远端地址的 SSH/WebIDE 会话也不可达，因此本节不把历史组件级或
算子级结果记为当前 wheel 的端到端通过。

## 2. 支持边界

| 项目 | 当前支持 | 处理方式 |
| --- | --- | --- |
| 硬件 | CSA/HCA：Blackwell `sm_100a/sm_103a`；MoE front：`sm_103a` | 首次执行前强校验 |
| 并行 | attention adapter：TP1、单进程；MoE front：TP1 + EP 多 rank（MegaMoE-SE） | attention 与 MoE 分别独立选路 |
| KV cache | FP8 | 非 FP8 初始化失败 |
| 层类型 | `compress_ratio == 4` 的 CSA 层（`DSV4_MEGA_CSA`）；`compress_ratio == 128` 的 HCA 层（`DSV4_MEGA_HCA`） | 按 ratio 分别挂 adapter |
| 模型几何 | DSV4-Pro（dim 7168）与 DSV4-Flash（dim 4096） | `GEOMETRY_BY_DIM` 按 `attn.dim` 选 profile，其他 dim 初始化失败 |
| 请求形态 | decode、`q_len == 1`、batch 1..128；MoE front 还要求 capacity-128 storage | 其余形态走现有路径 |
| 进程角色 | `DECODE` 和单卡 `PDFUSION` | 由 `forward_decode` 限制实际执行 |
| 开关 | `DSV4_MEGA_CSA=1` / `DSV4_MEGA_HCA=1` / `DSV4_MEGA_MOE_FRONT=1` | 各自默认关闭，模型构造期固定；MoE 使用独立 runtime |

下列场景保持现有实现：prefill、SWA-only、target verify (`q_len > 1`)、MTP、TP2/DP2。
MTP 是独立模型且当前 `compress_ratio == 0`，不会挂载 CSA adapter。

`is_decode_role=False` 同时覆盖 `PDFUSION` 和专用 PREFILL，框架目前没有更细的构造参数。
因此两个 Mega 开关都只应配置在 `DECODE/PDFUSION` 进程；误配到专用 PREFILL
不会执行 Mega decode，但会产生不必要的 fused-weight 重排和显存占用。

## 3. 已完成的框架适配

### 3.1 文件与职责

| 文件 | 修改 |
| --- | --- |
| `dsv4/transformer.py` | 解析开关；校验 FP8 KV/TP1；创建模型级 runtime；给 CSA 层挂 adapter |
| `dsv4/decode/forward.py` | 在生产 layer loop 前推进一次 Mega decode step |
| `dsv4/block.py` | 在 attention sublayer 入口选择完整 Mega 路径；FFN 前重新汇合 |
| `fp8/decode/mega_csa_weights.py` | 校验 checkpoint tensor 并构造算子要求的 TP1 fused layout |
| `fp8/decode/mega_csa_runtime.py` | 共享 workspace、logits、MQA schedule 和 RoPE table；校验并透传框架 slot tensor |
| `fp8/decode/mega_csa_adapter.py` | 绑定现有 cache/metadata，编排 Mega 算子、TopK、原生 FlashMLA 和 o-proj |
| `fp8/test/test_mega_csa_adapter.py` | 覆盖选路、PDFUSION、权重布局、ABI 和 runtime 生命周期 |
| `fp8/test/test_mega_csa_rtp_eager.py` | 用真实 `AttentionFP8`/`KVCache` 对照原 attention 子层，并覆盖 eager、graph 和 cache/state 正确性 |
| `fp8/decode/mega_hca_weights.py` | HCA 层 fused 布局：`front_fp8=[wq_a;wkv]`、`front_bf16=[comp_wkv;comp_wgate]`、`wq_b`，约 130 MiB/层 × 31 层 ≈ 4 GiB |
| `fp8/decode/mega_hca_adapter.py` | HCA 编排：front/WQ-B 两个融合 GEMM、CUDA `q_rmsnorm_rope_cuda_`、稠密 idx、原生 FlashMLA、共享 o-proj producer |
| `fp8/decode/mega_csa_runtime.py`（扩展） | 新增 HCA workspace 缓存与 HCA 三组 slot（HCA_STATE/HCA_KV/SWA_KV）int64 直传校验，`begin_decode`/rope 表与 CSA 共享 |
| `dsv4/block.py`（扩展） | `enable_mega_hca`（仅 ratio==128），`forward_decode` 统一 `_mega_csa_adapter or _mega_hca_adapter` 选路 |
| `fp8/test/test_mega_hca_adapter.py` | HCA 选路、双开关、权重布局、geometry/ABI、runtime slot 生命周期 |
| `fp8/test/test_mega_hca_rtp_eager.py` | 真实 `AttentionFP8(ratio=128)` 对照原 `_forward_decode_hca`：输出、边界压缩 HCA_KV、SWA、state 环、长上下文和 graph 正确性 |
| `mega_csa_weights.py`（Flash 化） | `CSAGeometry` profile（PRO/FLASH），打包形状全部由 profile 派生；模块级 Pro 常量保留为别名 |
| `mega_csa_adapter.py` / `mega_hca_adapter.py`（Flash 化） | `_validate_geometry` 按 `attn.dim` 选 profile 并 fail-fast；ABI 探针改为按本层几何的子集校验（extension 广告双形状） |
| `mega_csa_runtime.py`（Flash 化） | CSA/HCA workspace 尺寸按 profile 分配并以 dim 入 key；`num_hc_splits` 接受 hidden 宽度 |

### 3.2 权重

`MegaCSAWeights` 在模型初始化时从原 checkpoint tensor 构造以下连续布局：

```text
front_fp8 = [wq_a; wkv]
front_sf = [wq_a_scale; wkv_scale]
front_bf16 = [main_wkv; main_wgate; index_wkv; index_wgate; index_weight_proj]
wq_b_fp8 = [index_wq_b; main_wq_b]
wq_b_sf = [index_wq_b_scale; main_wq_b_scale]
```

FP8 权重和 UE8M0 scale 不做数值反量化/再量化。Indexer score 的两个归一化因子在初始化时
折入 `index_weight_proj`，与现有 `IndexerFP8` 语义一致。

当前每个 CSA 层约增加 158 MiB 连续权重副本（DSV4-Pro 的 30 个 CSA 层约 4.6 GiB）。它不影响单步
kernel 时间，但影响模型初始化和常驻显存；在保留普通 target-verify 路径时不能直接释放原权重。
后续可评估 loader 直接产出 fused layout，或者调整 kernel 接受分段权重，避免重复存储。

### 3.3 模型级 runtime

所有 CSA 层顺序复用同一批按 `(device, batch, split)` 缓存的 workspace，不按层重复分配。
runtime 还负责：

- 每个模型 decode step 只生成一次 MQA schedule；
- 在 WQ-B 提交前准备 schedule，保持 WQ-B 到 MQA 的 PDL 顺序；
- 校验框架五组 slot mapping 为连续 CUDA int64 tensor，并将原 tensor 直接传给算子；
- 保留 capture 期间生成的 schedule tensor，避免 graph 中悬空指针；
- 缓存从 `freqs_cis` 拆出的连续 cos/sin table。

`cuda_extension@e1d1c985` 已把 FP8 CSA 的五组 slot ABI 改为 int64，与 RTP metadata 对齐；
`cuda_extension@b93e0761` 把 HCA 的三组 slot（state/window/compressed destinations）同样升级
为 int64，并新增 `geometry_hca()` 供 fail-fast ABI 探针（HCA front 的 PDL 按算子契约保持关闭）。
runtime 不再分配 int32 mirror，不执行 `copy_`，也不按 eager/graph metadata 缓存 slot 副本；
CUDA Graph 捕获期间直接使用框架 tensor 的稳定地址。position、block table、context length 和
schedule metadata 等其他 ABI 均未扩大为 int64。

### 3.4 Cache 与 FlashMLA

没有增加通用 cache ABI。adapter 直接使用现有：

```text
pool_block_tables
pool_write_slot_mappings
compressor_state_slot_mappings
compressed_lens
topk_buffer_compressed
position_ids / position_ids_long
swa_global_slots
```

`entries_per_block` 和 `block_stride_bytes` 继续从 typed pool 的现有 view/stride 推导。

FlashMLA wrapper 没有修改，也不依赖 Wuda 的改造版 FlashMLA。adapter 在写 cache 前检查
现有 FlashMLA metadata 和 backend，再通过 `AttentionFP8._forward_decode_compressed` 调用
RTP 当前原生 FlashMLA wheel。进入 Mega 且发生 cache write 后，任何错误直接上抛，禁止回退
普通 attention，避免同一步重复写 cache。

Wuda `origin/main@6818258` 新增的 MLA output inverse-RoPE + FP8 quant CUDA producer 已迁入
`rtp-kernel`。Mega runtime 为它提供模型级复用的 graph-stable FP8/scale workspace；adapter 直接
传框架 int64 position 和已有的 FP32 cos/sin table，不再先执行 `freqs_cis.index_select`。producer
输出继续交给 RTP 现有 `_wo_a_einsum_from_fp8` 和 `wo_b`。普通 attention 路径仍使用原 Triton
producer，没有修改通用 output-projection 选路。

### 3.5 普通路径影响

开关关闭时不构造 fused weights、runtime 或 workspace，也不新增 CUDA kernel。
普通路径保留原有 tensor 和 cache ABI。代码层只增加一次 model-step runtime presence check，以及
每层一次 `adapter is not None` 的 Python 分支；是否可测必须由 normal FP8 A/B 给出，不能只凭
静态分析宣称零下降。

## 4. 算子与制品状态

CUDA Extension 已完成 Wuda 最新 TP1（不含 TPDP）迁移并推送：

```text
repo:   git@gitlab.alibaba-inc.com:foundation_models/cuda_extension.git
branch: dsv4_megakernel
base:   origin/main@3bc0ca4
source: Wuda origin/main@6818258 + origin/flash@ce0b82b（Flash CSA 几何）
commit: 9f4c3fe fix(dsv4): accept 64-head query rows in fused/standalone query RMS+RoPE
        aac0948 feat(dsv4): compile CSA ops for Flash geometry
        b93e0761 feat(dsv4): consume int64 slots in HCA decode ops
        e1d1c985 feat(dsv4): fuse MLA output inverse RoPE quant
```

Flash 支持的实现方式：CSA 链（front/wq_b/hc_reduce）在同一源码上以
`DSV4_FLASH_CSA` 编译出第二组 TU（`*_flash.cu`），所有受宏影响的 namespace/kernel
符号重命名以避免 ODR 合并；pybind 注册 `*_flash` 入口，python wrapper 按输入形状
一行 dispatch。`geometry_csa()`/`geometry_hca()` 同时广告两组形状。随迁的几何无关
优化：wq_b BM16 小批模板（M<=16）、qnorm warp 数模板化、mqa fp4/fp8 与 standalone
`query_rms_rope_out` 的 `output_heads` 运行时化（64/128）。

wheel 由 `build.py` 本地构建（见 §8.2），文件名含 git hash 与构建时间戳，包含：

```text
rtp_kernel/dsv4_mega.py
rtp_ops_dsv4_mega.cpython-310-x86_64-linux-gnu.so
```

RTP 公共 CUDA13 lock 解析到的 `rtp-kernel 0.1.0+cu13.*` 官方 wheel 尚不含 `dsv4_mega`。
当前验证通过未跟踪的本地 lock 补丁（指向本地构建 wheel 的 `file://` 行 + sha256）让
Bazel 解析，没有替换 Bazel external cache。必须发布 wheel 后再更新公共 requirements 和
lock，不能把本地绝对路径或临时 URL 写进提交。

CSA adapter 另外依赖当前 DeepGEMM 的：

```text
tf32_hc_prenorm_gemm
get_paged_mqa_logits_metadata
get_num_sms
```

HCA adapter 只依赖其中的 `tf32_hc_prenorm_gemm`（无 MQA schedule）。

首次真实执行会同时检查 GPU capability、`rtp_kernel.dsv4_mega` 函数签名和固定 geometry，
避免“有同名旧符号但 ABI 不兼容”时进入 cache write。

## 5. 已完成验证

以下 CPU/静态回归已通过：

```text
//rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_csa_adapter
//rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_csa_overlap
//rtp_llm/models_py/modules/dsv4/fp8/test:test_decode_topk_length
//rtp_llm/models_py/modules/dsv4/decode/test:decode_fmha_impl_test
```

以下完整编译已通过：

```bash
bazelisk build //rtp_llm:rtp_llm \
  --verbose_failures \
  --config=cuda13 \
  --test_output=errors \
  --test_env="LOG_LEVEL=INFO" \
  --jobs=64
```

adapter ABI 检查（函数签名 + geometry 探针）已通过：Pro geometry 为 main `65536`、
index `8192`、merged `73728`、main heads `128`、index heads `64`，slot ABI 为 int64；
Flash 化后探针改为按本层几何的子集校验（见 §3.1）。

新增 SM100 单卡测试：

```text
//rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_csa_rtp_eager
```

该测试显式固定 Wuda `origin/config.json` 中的 DSV4 Pro attention geometry：

```text
dim=7168, n_heads=128, q_lora_rank=1536
head_dim=512, rope_head_dim=64
o_groups=16, o_lora_rank=1024
window_size=128, compress_ratio=4 (CSA)
index_n_heads=64, index_head_dim=128, index_topk=1024
original_seq_len=65536, max_seq_len=65536
rope_theta=10000, rope_factor=16, beta_fast=32, beta_slow=1
compress_rope_theta=160000, hc_mult=4, hc_sinkhorn_iters=20
FP8 indexer, TP1, RTP persistent TopK, RTP wo_a/wo_b, official FlashMLA
```

Mega 和 reference 都调用 RTP 现有 persistent TopK；没有迁移或选择 Wuda TopK。

测试使用一个真实 `AttentionFP8` 层、确定性合成权重和两套相同初态的 RTP pybind `KVCache`。
reference 严格执行 `Block.forward_decode` 的原 attention 分支：

```text
attn_hc.pre -> attn_norm -> AttentionFP8.forward_decode -> attn_hc.post
```

Mega 与 reference 分别写独立 cache，连续执行 position `0..3` 到首个 CSA compression boundary。
结果如下：

| 对照项 | `calc_diff` / 结果 | 门限 |
| --- | ---: | ---: |
| 最终 attention sublayer 输出 | `1.135427e-05` | `< 1e-3` |
| CSA KV（解量化） | `1.261505e-05` | `< 1e-3` |
| Indexer KV（解量化） | `3.661147e-04` | `< 1e-3` |
| SWA KV（解量化） | `4.985002e-07` | `< 1e-3` |
| CSA state | `5.116385e-11` | `< 1e-4` |
| Indexer state | `5.772682e-11` | `< 1e-4` |
| TopK | int32 全量一致 | 精确一致 |
| CUDA Graph replay | bitwise 一致 | 精确一致 |

另在 position `4095` 预填充 1024 个随机有效 FP8 packed CSA/Indexer cache entry，从 1024 个
候选中选择 Top-1024：Mega/reference 有效 TopK overlap 为 `1024/1024`，最终输出
`calc_diff=3.094866e-09`。

### 2026-08-18：Mega HCA TP1 接入与验证

HCA 接入提交：开源 `c558d9b27`（本仓）+ `cuda_extension@b93e0761`。关键几何均取自
`DSV4CacheConfigHelper.cc`：HCA_KV 为 `tokens_per_block/128 = 2` entries/block；HCA_STATE
ring 为 `computeStateRing(128, kHcaOverlap=0, gen)`，非 MTP 时恰为 128（注意 `kHcaOverlap`
是 0，不是 CSA 的 1）；state 行为 `kv(512)|gate(512)` 交错 fp32，算子以两个 stride-1024
view 直接写框架池。

新增测试：

```text
//rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_hca_adapter
//rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_hca_rtp_eager
```

单层对照结果（真实 `AttentionFP8(compress_ratio=128, indexer=None)`，对照原
`_forward_decode_hca` 分支，B300/sm_103a）：

| 对照项 | `calc_diff` / 结果 | 门限 |
| --- | ---: | ---: |
| 最终 attention sublayer 输出（pos 0..3） | `2.140570e-05` | `< 1e-3` |
| 边界压缩步输出（pos 127，随机 state 环） | `7.721056e-07` | `< 1e-3` |
| 边界压缩写入 HCA_KV（解量化） | `1.533674e-05` | `< 1e-3` |
| SWA KV（解量化） | `0.0`（bitwise） | `< 1e-3` |
| HCA state 环 | `1.1e-06` | `< 1e-4` |
| 长上下文（pos 4095，随机 cache+state） | `4.038074e-06` | `< 1e-3` |
| CUDA Graph replay | bitwise 一致 | 精确一致 |

CSA 回归在新 wheel 与共享 runtime/block/transformer 改动下重跑，数值与 2026-08-15 基线
逐位一致。

### 2026-08-18：裁层 DSV4-Pro 本地 serving 端到端

用 `docs/dsv4_mega_e2e/truncate_dsv4_pro.py` 从全量 DSV4-Pro checkpoint（内部
NAS 个人共享目录 `/mnt/nas1/nanjun.cp/DeepSeek-V4-Pro`）裁出 4 层
（`compress_ratios=[128,128,4,128]`，59 GB，`num_nextn_predict_layers=0`，需带 `encoding/`）。
单卡 `rtp_llm.start_server`（BF16 act、FP8 KV、`seq_size_per_block=256`）baseline 与
Mega（CSA+HCA 双开）各完成 3 条 greedy 请求（含跨 128-token 压缩边界的 200-token 生成），
无 crash/NaN。token 前缀 7~12 个一致后分岔；logits 对照 `calc_diff=5.6e-04/7.5e-04`、
`top1_same=1`、baseline top1 margin 仅 `0.0036/0.0040`——4 层裁层模型 argmax 病态敏感，
分岔不构成 Mega 缺陷证据；正确性定论需健康模型（见下）。

### 2026-08-19/20：Flash 双几何与全量 Flash 端到端

extension 侧（`aac0948` + `9f4c3fe`）：

- 新增 `tests/test_dsv4_mega_flash_csa.py`：hc_reduce（DIM 4096）、front（K 4096/N 4160）、
  wq_b（K 1024/64 heads，覆盖 BM16 与全部 32-row 模板）对照 torch 参考全过
  （GEMM `calc_diff < 1e-5`，fp4 链 byte-match）；另含 64-head 奇数 batch 的
  `query_rms_rope_out` 回归用例。
- Pro CSA 6/6、HCA 4/4 pytest 回归全绿（含 BM16 新路径）。
- 修复：`output_heads` 运行时化漏掉的三处 128-head 行数前置校验
  （mqa fp8/fp4 `numel % (128*512)`、standalone query_rms_rope 的 host 校验与
  kernel `>>7`）。Pro 偶数头批次永远不触发，Flash 首步 decode 即 abort；`9f4c3fe`
  修复并补测试。

RTP 侧（`5a393bedda` CSA、`da03d2b19c` HCA）：bazel
`test_mega_{csa,hca}_rtp_eager` + `test_mega_{csa,hca}_adapter` 全绿（Pro 回归口径）。

全量 Flash 端到端（`/mnt/nas1/hf/DeepSeek-V4-Flash-0731`，43 层 156 GB，单卡，
`max_seq_len 4096`）：baseline 与 Mega（CSA+HCA 双开，SWA-only 前两层走原生路径）
各完成 3 条 greedy 请求。两侧输出均语言连贯且关键答案一致（"Paris"、"2+2=4"）；
200-token 长生成前约 160 字符逐字相同后在近平局 token 处分岔，之后各自连贯。
0/3 文本逐字一致——与框架 smoke 对 cp2/cp4/tp1 各用 golden 的既有现象同类，
文本级验收应采用 per-配置 golden；logits 级定量对照见下节。

### 2026-08-20：现成 smoke golden 用例 × Mega 三方对照与 logits 定量

用框架自带 smoke（`q_r_v4_flash_sm100_arm.json`：5 条 query 带 golden——2 条
greedy、1 条 4261-token 长上下文、2 条 507 错误路径）在本机对同一 checkpoint 分别
跑 baseline 与 Mega（args 完全一致，见 §8.4 复现要点）。golden 生成环境为
ARM + 7 月 Flash 快照，本机为 x86 + `-0731` 快照：

| query | golden | baseline（本机） | Mega（本机） |
| --- | --- | --- | --- |
| Paris | `...is Paris.` | `...is **Paris**.` | 与 baseline 逐字一致 |
| 2+2= | `That's a simple...` | `2 + 2 = **4**.` | `2 + 2 = 4.`（尾 token 近平局） |
| 4261-token 长上下文 | `DSV4_TP1_LONG_CONTEXT_OK` | 同 | **三方逐字全等** |
| 507 错误路径 x2 | — | 通过 | 通过 |

最重的长上下文 case（长 prefill -> Mega CSA+HCA decode -> 对抗式指令跟随）三方
全等；短 case 的 golden 漂移连 baseline 也复现不了（环境/快照差异），符合框架
per-环境 golden 的既有认知。

logits 级定量（`run_e2e_logits.py`，服务器返回**最后一步** logits，只统计两侧
生成前缀一致的有效样本）：

- prefill / 底噪：`max_new_tokens=1` 时（返回值即 prefill 输出）4 条 prompt
  baseline vs mega 与 baseline vs baseline 复跑全部 `calc_diff=0.0`（bitwise）——
  prefill 运行间确定，且 Mega 开关对 prefill 零扰动。
- decode 第 8 步（经 7 步 Mega decode 累积，有效样本 3/4）：`calc_diff`
  `6.7e-04`~`2.5e-03`，top1 全部一致（margin 0.13~10.7），低于框架 smoke 数值档
  `isclose(1e-2)` 一个量级。

### 2026-08-22：compressor state 正确性修复与精度对齐

本轮对应 RTP `c24dbcde6..4ca774e0c` 与 `cuda_extension@360b83f..2d6261a`，主要完成
两类收口：

1. **Compressor state pool 正确性**：框架为 CSA/Indexer state pool 每个 request 固定
   分配两个物理 block，并通过 256-logical-token 页的 block table 映射；每个物理 block
   内仍只保存 8 个 state ring entry。旧 Mega 实现假设 compressor 的 8-token 窗口都属于
   当前物理 block，因此窗口跨过 256-token 页边界时会读错 state。RTP 现将 CSA_STATE、
   INDEXER_STATE block table、token-to-request 映射和每页 token 数传给 opB/MQA，extension
   对窗口中的每个 logical position 分别解析物理 block；页边界与非连续 destination 测试已补齐。
2. **精度语义对齐**：CUDA inverse-RoPE + FP8 quant 保持 FP32 RoPE 结果到量化前，并按
   Triton 语义处理 FP8 scale 下限；CSA Indexer/main compressor 与 HCA compressor 对齐
   RMSNorm、RoPE 和 BF16 rounding 的先后顺序及 scale floor；HCA query RMSNorm+RoPE 切换
   到 CUDA 实现并复用模型级 RoPE table cache；CSA/HCA mHC tail 在 `--use_fast_math` 下恢复
   normal-precision `expf` 语义，并按 TileLang 的顺序做 Sinkhorn warp 规约。相应的
   CUDA-vs-Triton 单测和 forced-prefix baseline/HCA/CSA/full-Mega 验证脚本已加入。

mHC 的 `post`/`comb` 系数仍不与源路径逐位一致，原因已经定位在生成这两个系数之前的
24 维 `mix` split-K 归约：两边都由同一个 DeepGEMM GEMM 产生 split partial，但源路径的
TileLang `pre_big_fuse` 对每个 `mix` 元素按 split 顺序串行累加，Mega 的
`hc_reduce_fuse_out` 则让 warp lanes 分摊 partial 后用 shuffle tree 归约。两种 FP32 求和的
结合顺序不同，因此得到略有差异的 `mix`；`post` 的 sigmoid 和 `comb` 的 20 轮 Sinkhorn
继承该差异。此前已经对齐 normal-precision exp 和 Sinkhorn shuffle 顺序，剩余差异不来自
这些计算，也不来自 mHC post 算子本身：Mega 与源路径最终调用的是同一个
`block.attn_hc.post`。

针对 `2+2=` case 的 block-stage 对拍进一步确认了边界：除 mHC pre 的 `attn_in` 存在低于
噪声门限的浮点尾差外，Q/KV、attention、inverse-RoPE/quant 和 O-proj 链路均对齐，送入
mHC post 的 `attn_out` 完全一致（CSA-only 首个 CSA block 与 HCA-only 首个 HCA block 的
`max_abs` 均为 0）；差异第一次出现在 mHC post 输出 `attn_residual`，对应 `max_abs` 分别为
`4.882812e-04` 和 `1.953125e-03`。这说明 attention 主体没有引入差异，`attn_residual` 的
差异仅来自 Mega mHC pre 生成的 `post`/`comb` 系数。若要 bitwise 对齐，需要把 Mega 的
mix 归约重写为 TileLang 的串行 split 累加，会牺牲当前归约并行度且没有正确性收益；现有
误差在验收门限内，因此本轮保留实现，也不将其列为剩余缺口。

### 2026-08-25：8EP 无 MTP 固定 batch 单层耗时

测试配置为 8EP、无 MTP、固定每 rank decode batch。`B/M` 分别表示 baseline 和
CSA+HCA Mega。Attention 耗时为 timeline 中对应 CSA/HCA attention sublayer 的单层均值。
MoE 耗时按 batch 归一化，取 4 个 context、baseline/Mega 以及 CSA/HCA 层中从 attention
结束到下一层 attention 开始区间的等权平均。

| Batch/rank | Context | CSA Attn B/M (us) | CSA 加速比 | HCA Attn B/M (us) | HCA 加速比 | MoE (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8K | 171.1 / 111.9 | 1.528x | 128.1 / 88.4 | 1.448x | 140.8 |
| 32 | 32K | 202.0 / 121.1 | 1.668x | 131.9 / 90.5 | 1.458x | 140.8 |
| 32 | 64K | 214.5 / 127.0 | 1.689x | 134.0 / 92.5 | 1.448x | 140.8 |
| 32 | 128K | 240.8 / 157.5 | 1.529x | 138.0 / 96.0 | 1.438x | 140.8 |
| 64 | 8K | 199.1 / 114.9 | 1.733x | 129.5 / 82.9 | 1.563x | 162.4 |
| 64 | 32K | 217.6 / 134.5 | 1.618x | 139.3 / 97.9 | 1.422x | 162.4 |
| 64 | 64K | 236.3 / 155.6 | 1.519x | 142.4 / 100.9 | 1.410x | 162.4 |
| 64 | 128K | 263.3 / 183.7 | 1.434x | 146.7 / 104.4 | 1.406x | 162.4 |
| 128 | 8K | 196.0 / 122.6 | 1.599x | 138.1 / 90.8 | 1.521x | 177.4 |
| 128 | 32K | 223.6 / 153.1 | 1.461x | 142.0 / 100.7 | 1.410x | 177.4 |
| 128 | 64K | 253.6 / 186.0 | 1.363x | 147.8 / 107.2 | 1.379x | 177.4 |
| 128 | 128K | 313.7 / 230.4 | 1.361x | 160.9 / 114.9 | 1.400x | 177.4 |

### 2026-08-25：B300 四-kernel MoE front 生产接入

`Flash_DeepSeek_V4_Pro` 的 Mega-MoE provider 已把 decode FFN front 收敛为以下四个
有序 kernel，支持 logical `1 <= M <= 128`，workspace 保持 capacity 128：

```text
1. sm100_tf32_hc_prenorm_gemm_impl<24,28672,64,32,64,35,128,12,128,128>
2. mhc_pre_epilogue_kernel<MoeFrontCollapseNormPolicy>
3. moe_front_router_gemm_kernel<MoeFrontRouterExactPolicy|MoeFrontRouterDynamicPolicy>
4. moe_front_route_topk_kernel<MoeFrontRouteExactPolicy|MoeFrontRouteDynamicPolicy>
```

其中第一个 kernel 使用 pinned official DeepGEMM mHC 实现；第二个 kernel 完成 split-K
reduce、input RMSNorm、collapse、learned FFN RMSNorm，并从 BF16 publication 之前的 FP32
collapse accumulator 计算 FP32 SSQ。第三个 kernel 使用 BF16 Router 输入/权重和 FP32
accumulation，同时写 E4M3 activation、packed UE8M0 scale、post/comb；第四个 kernel 完成
learned V4 TopK-6。HashMoE 保持独立 checkpoint-table route publication。

待发布的 CUDA13 `rtp-kernel.dsv4_mega` ABI 通过 graph-stable
`Dsv4MoeFrontPlan(hidden_states, hc_fn, logical_m)` 暴露生产 ABI，learned/Hash 分别调用
`run_learned_out`/`run_hash_out`。Plan 绑定输入与权重指针，所有输出 workspace 由 caller
持有；front 按官方 MegaMoE buffer ABI 发布，随后继续调用现有
`deep_gemm.fp8_fp4_mega_moe`，最后进入 mHC post。provider 还必须通过
`geometry_moe_front()` 报告 `kernel_contract_version=1`；缺少该握手的旧 wheel 会被拒绝，
避免同一 Python 签名静默落到旧 `hc_gemm_splitk_kernel`。

SM103（148 SM、CUDA 13.0、PyTorch 2.11.0+cu130、DeepGEMM
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`）单独 front 结果如下。口径为一次 warmed
CUDA Graph replay，从第一个 production CUDA kernel start 到第四个 kernel end 的 Kineto
envelope；cold-L2 在同 stream 先写 8 GiB，flush 本身不计时。每点 30 次 warm、7 次 cold：

| M | warm graph envelope | cold-L2 graph envelope |
| ---: | ---: | ---: |
| 16 | `18.848 us` | `22.367 us` |
| 32 | `19.022 us` | `22.750 us` |
| 64 | `19.198 us` | `23.328 us` |
| 96 | `19.247 us` | `24.063 us` |
| 128 | `19.840 us` | `24.447 us` |

所有 M 的 TopK index 精确一致，浮点差约 `2e-14`；30 个 warm sample 的 production
kernel count 全部为 4，M128 Perfetto 也恰好显示上述四个 semantic kernel。结果来自 RTP
`MegaMoEFrontAdapter.forward_ffn_sublayer` 对 provider `moe_front_enqueue` 合约的真实调用，
不是绕过 adapter 的 provider-only 测量。最终 adapter SHA256 为
`4a4943f754278ea8ed09ff6115880e77b6ae5ecdebad3fed94a2597dc0613486`，结果 JSON SHA256
为 `ee0a9fc9803e06f8c29ccf2e2de4da62047701f75ce427888066bb4929a47199`，M128 Perfetto
SHA256 为 `18957fc602c99ca04690192847b269fe4c95304606bc8f0b0bad4dfb092a5c19`。该表只覆盖
MoE front，不能当作 expert core、EP communication、mHC post 或完整 DecoderLayer 延迟。

### 2026-08-25：官方 MegaMoE-SE 组件级 A/B（非 RTP wheel e2e）

本地保留了一组独立于 RTP adapter 的官方 MegaMoE graph A/B，控制变量为同一随机种子
`20260825`、同一 shape `(tokens=64,max_tokens=128,experts=384,topk=6)`、同一
DeepGEMM commit `559d79fb6994a58b8a15b4b93bf13ccc16edf247`，唯一变量是
`num_shared_experts=0/1`。设备 capability 由 PyTorch 记录为 `(10,3)`，每个变体 4 rank、
30 warm + 10 cold samples，cold 前同 stream flush 8 GiB：

| 变体 | 4-rank envelope median | rank-max envelope median | warm CUDA-event median | cold CUDA-event median |
| --- | ---: | ---: | ---: | ---: |
| no-SE | 661.168 us | 678.111 us | 556.640 us | 693.440 us |
| MegaMoE-SE | 661.550 us | 682.622 us | 571.088 us | 696.128 us |

在同一份 internal phase trace 中，Dispatch duration 为：no-SE
`36.896/36.992/151.360/88.288 us`、SE `41.824/61.856/90.656/86.240 us`（rank 0..3）。
shared expert 的 L1/L2 阶段会占用资源并改变 dispatch overlap，但该 workload 的总 envelope
没有稳定拉开；rank 2 的 no-SE dispatch 尾延迟反而更大。原始证据为
`artifacts/megamoe-se-ab-4xb300-20260825/summary.json`（SHA256
`fcfe0c06679461f7b03a6cbc7729070aa30b1ba44bd5416ac9eab229ebcd22c4`）和
`internal_phase_ab_summary.json`（SHA256
`72012978b3bd493a571a11f9356bc71af21aba35fe6c260e28d0a267e02835f4`）。

这组数据只证明官方 MegaMoE 内部的 SE 资源/overlap 行为，不证明 RTP 的
`Dsv4MoeFrontPlan`、CSA/HCA、expert core 和完整 decoder layer 已经通过 wheel e2e；后者仍以
§6 的 EP4 serving 和 logits/layer trajectory 为准。

RTP 生产接入由以下部分组成：

1. `moe/native_front.py` 负责完整 Plan ABI/geometry 校验、模型级 capacity-128 workspace、
   pointer-bound Plan cache，以及 learned/Hash 的显式调用；`collapse_ssq` 固定为 FP32；
2. `Block.forward_decode` 在支持条件满足时把 attention residual 直接交给 native front，跳过
   原 `ffn_hc.pre + ffn_norm + Router F.linear + gate-pack`；
3. front 零拷贝写现有 MegaMoE-SE symmetric buffer 的
   `x/x_sf/shared_l1_acts_sf/topk_idx/topk_weights`，expert core 通过
   `forward_prepacked` 直接消费，不增加 publication copy kernel；symmetric buffer 可拥有多于
   128 行，但 Plan ABI 要求精确 `[128,...]`，因此 adapter 为四个 row-major publication
   tensor 创建同 data pointer 的精确 capacity view，不产生 copy 或第五个 kernel；
4. RTP 的 logical residual 是 `[M,1,4,7168]`，provider Plan 要求固定
   `[128,4,7168]`。decode 入口把原 HC `repeat` publication 改为写模型级 capacity-128
   buffer，并一次 staging capacity-128 `input_ids`；此后 HC post 原地复用该 residual
   storage。adapter 创建同指针 `as_strided` capacity view；外部调用若没有该容量则回退旧
   路径，禁止在四-kernel front 内补 copy；
5. HashMoE layers 使用 INT32 `tid2eid` 和 capacity-128 INT64 `input_ids`；learned layers 使用
   FP32 correction bias。prefill、MTP、`M > 128`、非 Pro geometry、非 MegaMoE-SE、短 storage
   均保留原实现；
6. 模型初始化开关为 `DSV4_MEGA_MOE_FRONT=1`，并必须同时设置
   `DSV4_USE_MEGA_MOE_SE=1` 选择带 shared expert 的 Mega strategy。当前只允许
   `D=7168/E=384/TopK=6/hc=4/TP=1/EP>1`，运行时再通过
   `torch.cuda.get_device_capability()` 校验设备 capability 必须为 `(10,3)`。当前
   four-kernel C++ plan 只实现 SM103；SM100 仍仅适用于 CSA/HCA attention adapter。

对应单元测试为：

```bash
bazel test --config=cuda13 \
  //rtp_llm/models_py/modules/dsv4/test:test_mega_moe_native_front
```

该测试覆盖 ABI/geometry 拒绝、FP32 SSQ、短 storage 回退、capacity view 同指针、Plan
复用、learned/Hash 输出 buffer 绑定，以及 `Block.forward_decode` 不再调用旧 FFN front。
2026-08-25 在上述 4 卡机器的 `cuda:0` 上，使用匹配 provider contract 和本节
`native_front.py` 完成真实 adapter benchmark：通过 PyTorch CUDA runtime 得到的 capability 为
`(10,3)`，覆盖
logical `M=16/32/64/96/128`、正确性、warm/cold-L2 Graph envelope 和 M128 Perfetto。
仓库 Bazel target 已进入依赖解析，但该机器访问 `bazel_skylib-1.0.2.tar.gz` 的内部 OSS
镜像超时，因此不把本轮记为 Bazel test pass。
正式发布仍需把匹配 ABI/contract 的 wheel 固定到构建依赖，并完成真实 EP4 CUDA Graph
和完整 MoE block 数值回归；单 rank adapter 四-kernel Perfetto 已完成。

## 6. 端到端剩余缺口

按阻塞顺序还需要：

1. 发布远端 `dsv4_megakernel@c81d23d` 对应的 CUDA13 x86_64 wheel，更新开源/内源
   实际使用的依赖入口和 lock；本地 `dc880c9` 只移除了 benchmark 的 inventory-tool
   查询，不改变 wheel 源文件或 `DSV4_MOE_FRONT_SOURCE_SHA256`；
2. 增加由真实 `KVCacheManager` 创建 typed pools/block tables 的集成测试，替代手工 pool
   fixture（本地 serving e2e 已实际走真实 allocator，但缺 bazel 内可回归的形式）；
3. ~~校验 normal prefill -> Mega decode~~ 已在裁层 Pro 与全量 Flash serving 中覆盖
   （target verify / MTP 场景仍未覆盖）；
4. 整模型正确性收口：为 Mega 配置生成 per-配置 golden（框架 smoke 惯例），并在健康模型
   上完成 logits 级对照（Flash 对照排队中）；建议同时把 4 层 Pro 裁层 checkpoint 上传 NAS
   并新增 `v4_pro_4layer_tp1` / `..._mega` smoke case；
5. 测量开关关闭时普通 FP8 整模型路径，确认新增 Python 分支不可测；
6. 对 normal FP8 与 Mega FP8 做真实模型、代表性长上下文和完整 batch grid 性能 A/B。
7. 发布包含 `Dsv4MoeFrontPlan` 与 `kernel_contract_version=1` 的 CUDA13 wheel，并完成
   HashMoE 与 EP4 多 rank 的完整 MoE block 验证；learned routing 的单 rank CUDA Graph、
   四-kernel Perfetto timeline 已完成。

性能报告至少应单列：

- 框架 int64 slot 直传，并确认 timeline 中没有隐式 conversion/copy；
- mHC pre 到 front、WQ-B 到 MQA 的 PDL 收益；
- MQA schedule 生成；
- TopK + 原生 FlashMLA；
- 完整 attention sublayer；
- 开关关闭的普通 FP8 路径；
- eager 与 CUDA Graph；
- batch 1/8/16/32/64/128 和代表性 context length。

## 7. 内源合入方案

目标内源分支为 `develop/wangyin_ds_v4_20260424`。在开源提交稳定后（迁移清单现含 CSA 与
HCA 两组 adapter/runtime/weights/测试文件）：

1. 将目标内源 worktree 对齐远端分支，保留现有用户修改和 gitlink；
2. 迁移本分支的 adapter、runtime、weights、选路及测试文档改动，不迁移 Wuda TPDP 或改造版
   FlashMLA 逻辑；
3. 新 wheel 发布后，同时更新内源 CUDA13 requirements lock 和实际 Bazel 依赖选择；
4. 先跑与开源相同的 CPU tests 和 `//rtp_llm:rtp_llm` 完整编译；
5. 再在内源服务配置中只对 TP1 FP8 `DECODE/PDFUSION` 打开开关，按 `DSV4_MEGA_CSA` →
   `DSV4_MEGA_HCA` → 双开的顺序分阶段验证；双开后 Mega 覆盖全部 61 个 attention 层。

HCA 已按同样的“完整 sublayer adapter”模式接入（`MegaHCAAdapter`，独立 geometry 检查）。
SWA-only、prefill、TP2/DP2 与 FlashMLA 通用接口仍不修改；若后续接入这些场景，应分别
新增受支持的完整 sublayer adapter，不能放宽现有 CSA/HCA TP1 adapter 的 geometry 检查。

## 8. 开发操作手册（分支 / 编译 / 运行 / 测试）

以下为本文档所有验证实际使用的流程，可在任一台 SM100/SM103（CUDA 13）内部开发机
上复现。`<work>` 代指你的工作目录。

### 8.1 仓库与分支

| 仓库 | 地址 | 分支 | 角色 |
| --- | --- | --- | --- |
| Wuda（算子上游） | `git@github.com:guluguluhhhh/wuda.git` | `main`（Pro TP1）、`flash`（Flash CSA 几何源，`ce0b82b`） | 只读迁移源，不直接部署 |
| cuda_extension | `git@gitlab.alibaba-inc.com:foundation_models/cuda_extension.git` | `dsv4_megakernel` | Mega 算子生产载体，出 `rtp-kernel` wheel |
| RTP 开源 fork | `git@github.com:guluguluhhhh/rtp-llm.git` | `dsv4-mega` | 框架适配（adapter/runtime/weights/测试/本文档） |
| RTP 内源 | gitlab `foundation_models/RTP-LLM` | `develop/wangyin_ds_v4_20260424` | 内源载体；子模块 `github-opensource` 指向上一行的 fork 分支 |

检出：

```bash
cd <work>
git clone -b dsv4_megakernel git@gitlab.alibaba-inc.com:foundation_models/cuda_extension.git
git clone <内源 RTP-LLM 地址> RTP-LLM && cd RTP-LLM
git checkout develop/wangyin_ds_v4_20260424
git submodule update --init github-opensource     # 或按 .gitmodules 换 fork 源后 checkout dsv4-mega
scripts/create_symlinks.sh
```

建议对内源仓另建 `git worktree`（例如 `.worktrees/dsv4-mega`）专用于 Bazel GPU
测试，主树做提交，测试前把改动文件同步进 worktree 同名路径，以保住 Bazel 缓存。

### 8.2 编译

CUDA Extension：准备 python3.10 venv 并安装 torch cu130 与
`cuda_extension/requirements.txt`，然后

```bash
cd <work>/cuda_extension
RTP_KERNEL_COMMIT_ID=$(git rev-parse HEAD) \
RTP_KERNEL_BUILD_TIMESTAMP=$(date -u +%Y%m%d%H%M%S) \
WHEEL_CUDA_VERSION=cu130 \
python build.py          # 构建普通 rtp_ops + rtp_ops_dsv4_mega，产物在 dist/
pip install --force-reinstall --no-deps dist/rtp_kernel-*.whl
```

生产 RTP serving 不要设置 `DSV4_MEGA_ONLY=1`：该过滤只保留
`rtp_ops_dsv4_mega`，会遗漏现有 expert core/kvcache 所需的 `rtp_ops`，导致
`import rtp_kernel` 失败。四-kernel ABI 仍由同一全量 wheel 中的
`rtp_ops_dsv4_mega` 提供。

冒烟：`python -c "from rtp_kernel import dsv4_mega; print(dsv4_mega.geometry_csa())"`
应同时出现 Pro 与 `*_flash` 两组形状。

RTP Bazel 依赖本地 wheel：修改两树（主树与 worktree）
`internal_source/deps/requirements_lock_torch_gpu_cuda13.txt` 中 `rtp-kernel` 行为
`rtp-kernel @ file:///<work>/cuda_extension/dist/<wheel 文件名>` 并更新其
`--hash=sha256:`（`sha256sum dist/*.whl`）。该补丁 **不提交**；wheel 重建后
（文件名含时间戳）必须同步刷新。完整编译：

```bash
cd <内源仓 worktree>/github-opensource
bazelisk build //rtp_llm:rtp_llm --config=cuda13 --jobs=64 --verbose_failures
```

### 8.3 运行（本地 serving 端到端）

serving 需要一个能 `python -m rtp_llm.start_server` 的 venv：python3.10 + torch
cu130 + 按 CUDA13 lock 安装依赖（大部分包用 `pip install --no-deps` 以防解析器拖走
torch；必须包含本地构建的 `rtp-kernel` wheel、`flashinfer-python`、
`nvidia-cutlass-dsl` 与 DeepGEMM），并把开源树 `rtp_llm/` 放进 `PYTHONPATH`
或安装进 site-packages（后者改代码后需重新同步）。

可用 checkpoint（内部 NAS，路径以实际挂载为准）：

```text
/mnt/nas1/hf/DeepSeek-V4-Flash-0731             156 GB 全量 43 层，单卡可跑（NAS 冷读约 40 min）
/mnt/nas1/nanjun.cp/DeepSeek-V4-Pro             865 GB 全量 61 层（个人共享目录）
docs/dsv4_mega_e2e/truncate_dsv4_pro.py         由全量 Pro 自制 4/6 层单卡裁层 checkpoint
```

一键对照脚本在 `docs/dsv4_mega_e2e/`（配置全部走环境变量，见各脚本 docstring）：

```bash
cd docs/dsv4_mega_e2e
E2E_CKPT=<checkpoint 目录> E2E_GPU=<idx> python run_e2e_compare.py
E2E_CKPT=... python run_e2e_logits.py baseline|mega|compare   # logits 级三步式
E2E_CKPT=... ./watch_and_run_logits.sh                        # 轮询空卡自动跑三步
DSV4_PRO_SRC=<全量 Pro 目录> python truncate_dsv4_pro.py --layers 4 --out <目录>
```

上述默认脚本是单卡 CSA/HCA 对照，`EP=1` 时不会执行 MoE front。Pro/EP4
端到端必须明确使用四张卡和 native front 开关：

```bash
E2E_CKPT=<DeepSeek-V4-Pro 目录> \
E2E_GPU=0,1,2,3 E2E_EP_SIZE=4 E2E_WORLD_SIZE=4 E2E_LOCAL_WORLD_SIZE=4 \
E2E_MOE_FRONT=1 python run_e2e_compare.py
```

该命令的 baseline 关闭三组 Mega 开关，mega 轮同时打开 CSA、HCA 和
`DSV4_MEGA_MOE_FRONT`；脚本会在启动参数中保持 TP1/EP4/world4，并对相同 greedy
请求做 token 级结果比较。

脚本封装的关键运行要素（手工起 server 时同样必需）：
`MODEL_TYPE=deepseek_v4`、`CHECKPOINT_PATH`/`TOKENIZER_PATH`、`START_PORT`；
`--load_method scratch --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 256`
（必须为 128 的倍数且 >=128）；共享容器内 `/tmp/rtp-llm` 可能属他人，需预设 8 个 JIT
cache 环境变量（`FLASHINFER_WORKSPACE_BASE`、`DG_JIT_CACHE_DIR`、`TRTLLM_DG_CACHE_DIR`、
`TILELANG_CACHE_DIR`、`TORCH_EXTENSIONS_DIR`、`TVM_FFI_CACHE_DIR`、`CUTE_DSL_CACHE_DIR`、
`TRITON_CACHE_DIR`）到自有目录（compare 脚本已代管）；`DG_JIT_CPP_STANDARD=20`。
Mega 开关：`DSV4_MEGA_CSA=1`、`DSV4_MEGA_HCA=1`、
`DSV4_MEGA_MOE_FRONT=1`、`DSV4_USE_MEGA_MOE_SE=1`（默认全关，即 baseline）。
MoE front 还要求
`D=7168/E=384/TopK=6/hc=4/TP=1/EP>1` 和 PyTorch capability `(10,3)`。

### 8.4 测试

CUDA Extension（pytest，单卡 GPU，全套约 1 分钟）：

```bash
cd <work>/cuda_extension
CUDA_VISIBLE_DEVICES=<idx> python -m pytest \
  tests/test_dsv4_mega_front_gemm_csa.py tests/test_dsv4_mega_wq_b_csa.py \
  tests/test_dsv4_mega_hc_fused.py tests/test_dsv4_mega_mqa_logits.py \
  tests/test_dsv4_mega_idx_post.py tests/test_dsv4_mega_mla_o_quant.py \
  tests/test_dsv4_mega_front_gemm_hca.py tests/test_dsv4_mega_wq_b_hca.py \
  tests/test_dsv4_mega_hca_chain.py tests/test_dsv4_mega_state_pool.py \
  tests/test_dsv4_mega_flash_csa.py -x -q
# tests/test_dsv4_mega_hca_e2e.py 需要 RTP_OPENSOURCE_ROOT 指向开源树
```

RTP（bazel，GPU 目标在 worktree 跑）：

```bash
cd <内源仓 worktree>/github-opensource
bazelisk test --config=cuda13 --jobs=64 --test_output=summary \
  --test_env=CUDA_VISIBLE_DEVICES=<idx> --nocache_test_results \
  //rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_csa_adapter \
  //rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_csa_rtp_eager \
  //rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_hca_adapter \
  //rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_hca_rtp_eager
```

CPU/静态回归与端到端见第 5 节与 8.3。

复用现成 smoke golden 用例做 baseline/Mega 对照（§5 2026-08-20 的做法）：
`internal_source/rtp_llm/test/smoke/BUILD` 中的 `v4_flash_native_fp4_fp8_tp1_*`
处于注释状态且 args 已过时，本地启用时需要三处适配（均不提交）：

1. task json 的 `model_path`（`/mnt/nas1/hf/DeepSeek-V4-Flash` 在部分机器是指向
   他机 `/data1` 的断链）指到本机可用的 checkpoint；
2. `--seq_size_per_block 64 -> 256`（当前分支 C++ 断言要求 >=128 且 128 的倍数）、
   `--max_seq_len 512 -> 8192`（长上下文 query 有 4261 token）、补 `--fp8_kv_cache 1`；
3. bazel 命令加 §8.3 的 8 个 JIT cache `--test_env`（smoke 子进程同样受
   `/tmp/rtp-llm` 权限问题影响）。

Mega 轮在 target 的 `envs` 里加 `DSV4_MEGA_CSA=1`、`DSV4_MEGA_HCA=1`。golden 是
旧环境产物（见 §5），判据看两轮 actual 的互相对照（bazel testlogs 的
`test.outputs/outputs.zip` 里有每条 query 的 actual dump）；正式收编需按框架惯例
生成本环境 per-配置 golden。smoke 宏会自动注入 `DETERMINISTIC_GEMM=1` 与
`DSV4_INDEXER_TOPK_CANONICALIZE=1`。

### 8.5 2026-08-26 wheel 集成复核（SM103 / 4 卡）

本轮在 WebIDE 的四卡目标机上完成了 CUDA Extension wheel 的实际构建、安装和
ABI 冒烟。设备架构只通过 PyTorch `torch.cuda.get_device_capability()` 读取，四张卡
均返回 `(10, 3)`，CUDA runtime 为 13.0；没有使用 `nvidia-smi` 作为架构判断来源。

构建输入和产物证据：

```text
cuda_extension source: c81d23db7f10a7ad6bf00a9535d30e207fef66c4
embedded source sha256: 5d9d222c7cd32c0969b2ed6cca25ffd837c07eddc07e3d204221179e32221ee0
wheel tag: rtp-kernel-0.1.0+c81d23db7f10a7ad6bf00a9535d30e207fef66c4.20260826061454.cu130
target: sm_103a
modules: rtp_ops + rtp_ops_dsv4_mega
moe-front kernels: 4
geometry: hidden=7168, hc_mult=4, hc_width=24, experts=384, topk=6,
          max_m=128, scale_cols=56, collapse_ssq_bits=32
```

安装后 `rtp_ops`、`rtp_ops_dsv4_mega` 和 `rtp_kernel.dsv4_mega` 均可导入；
`geometry_moe_front()` / `build_info_moe_front()` 返回上述合约。生产运行必须安装
完整 wheel，不能使用 `DSV4_MEGA_ONLY=1` 过滤掉普通 `rtp_ops`。

E2E runner `docs/dsv4_mega_e2e/run_e2e_compare.py` 本轮补齐了四个可复现性要求：

* `E2E_TP_SIZE`、`E2E_DP_SIZE` 可配置，四卡 EP 使用 `TP=1, DP=4, EP=4`；
* child 和 health-check parent 都固定到 `E2E_SOURCE_ROOT`，清除旧 venv 的
  `PYTHONPATH`，避免加载不匹配的 `libth_transformer_config.so`；
* JIT cache、`watchdog` 等 serving 依赖由运行环境显式准备。
* 启动前检查 `E2E_CKPT` 是 populated snapshot 且含 `config.json`，避免空缓存目录
  触发长时间 server 启动后才在 tokenizer 阶段失败。

远端 staged checkout 在首次适配层收集时暴露了
`dsv4/transformer.py` 缺少 `typing.Tuple` 导入的问题；已补齐并重新同步。匹配
native ABI 的 WebIDE 环境中，修复后的结果为：

```text
test_mega_moe_native_front.py: 10 passed in 2.37s
test_mega_moe_input_packer.py + test_mega_moe_se_input_pack.py
  + test_mega_moe_gate_pack.py: 14 passed, 9 subtests passed in 4.98s
```

这些测试覆盖 native-front ABI 检查、workspace/plan 生命周期、稳定 128-row
buffer、shared-expert publication、输入 pack、SE pack 和 gate pack；它们不代替真实
权重上的 token 生成测试。

四卡 server 已通过 native ABI、EP 参数和启动链路检查，但本轮不能给出 token 级
baseline/Mega TPS 或 Perfetto 对比：指定的 checkpoint

```text
/mnt/fuse/.cache/models--deepseek-ai--DeepSeek-V4-Flash-0731/
snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
```

在目标机上是空目录（无 `config.json`、tokenizer 或权重）。因此服务最终在 tokenizer
初始化处停止，而不是在 MoE-front、CSA 或 HCA 算子处失败。重现真实端到端前，需把
`E2E_CKPT` 指向实际挂载的 DeepSeek-V4-Flash/Pro 快照；快照可见后直接运行：

```bash
E2E_CKPT=<实际快照> \
E2E_GPU=0,1,2,3 E2E_TP_SIZE=1 E2E_DP_SIZE=4 \
E2E_EP_SIZE=4 E2E_WORLD_SIZE=4 E2E_LOCAL_WORLD_SIZE=4 \
E2E_MOE_FRONT=1 python docs/dsv4_mega_e2e/run_e2e_compare.py
```

该命令会先跑关闭 Mega 开关的 baseline，再跑同时打开 CSA、HCA 和四-kernel
MoE-front 的 mega 轮，并对固定 greedy queries 做 token 级比较。当前已获得的性能
结论仍限于算子层 cold-L2 CUDA Graph envelope：四-kernel MoE-front 的既有结果为
`25.536 us` median，相对旧 RTP 六-kernel 路径 `32.2405 us`，约 `1.2626x`；本轮未
把空 checkpoint 的启动失败误报为端到端性能数据。
