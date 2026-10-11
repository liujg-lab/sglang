# SR Draft seed 与多步 KV 索引

`SRDraftMetadataWorkspace` 没有留在当前工作区。相对 `c223c433d`，Triton
多步 backend 仍生成完整 S 行，FlashInfer 和 NPU 也没有接到这条 metadata 路径。

A–D 消融没有执行。编写环境没有 CUDA、Triton 和模型权重。B=1 graph
接受长度是否回退仍然未知，不能把主机单测或语法检查写成消融通过。
实机按下面的表跑完之前，不要把 FlashInfer 或 NPU 接到这条路径上。

## B=1 graph 接受长度下降的四组消融

在独立目录准备版本，不修改运行中的项目或实验启动脚本，不增加生产开关：

| 版本 | 内容 |
|---|---|
| A | `c223c433d` 基线 |
| B | A + seed 行索引共享（仅 sr_tail_extend 的整理修改） |
| C | A + 显式 S 跨度/S−1 行；eager 仍采用原临时缓冲和 clone |
| D | 当前完整修改（含 eager 生命周期修正） |

Target 固定同一版本；Draft 每组显式设置各副本的 `PYTHONPATH`，记录实际
import 路径、提交和 diff。两端模型、tokenizer、dtype、K/S/W、RPD tau、
图片预处理、prompt、请求顺序及缓存初始化完全相同。先用短请求和 B=1
定位首次差异，再扩展 B=2/3/4、3→4 padding 与 eager。

仅在诊断副本中，将以下数据以请求 ID、轮次、step 对齐保存（不能只比较
最终文本或平均接受长度）。需要读回时同步并 clone，避免记录被下轮覆盖：

1. tree graph 前的 seed 概率/token/root、请求行、seq_lens、positions、
   mRoPE positions、实际 graph/eager admission。
2. 元数据生成后的每 step 有效 indices/indptr、映射及对应物理 KV。
3. 每 step 的 logits、top-k、parent、候选树；逐位或精确比较整数，
   浮点同时报告精确相等与最大差异，不能先用宽松容差掩盖 top-k 变化。
4. Target 收到的树、核验 logits、RPD 路径、停止截断和截断前接受长度。

若 indices 先不同，检查跨度、真实 stride、写入范围及 Triton 编译结果，
用完整 S 行作为正确性对照。若 indices 相同而 logits 先不同，继续定位
KV/位置/执行状态；不能直接认定少生成一行导致。若树完全相同而接受不同，
排查 Target 状态与核验；中间结果均相同才检查评测口径。

阶段同步只用于定位。性能验收删除诊断同步和记录，在正确性通过后预热，
新旧版交替三组、每组至少 500 轮，分别报告 B=1/2/3/4 的 graph/eager、
整轮延迟、吞吐和接受长度。当前文档不承诺加速或接受长度恢复。
消融要用独立目录里的副本，不在这棵工作区里重新接上 workspace。
