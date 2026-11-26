---
name: clear-tree-e2e-validation
description: Runs a real ClearTree remote end-to-end experiment from the user's current matrix. Use when the user explicitly asks to run or prepare ClearTree E2E on CUDA Draft and Ascend NPU Target. Concurrency, batch capture, request count, and switches come from that request.
disable-model-invocation: true
---

# ClearTree 真实端到端验证

流程是「建档 → 空闲保护与窄同步 → 按用户矩阵运行 → 离线验收 → 清理」。只跑用户指定的并发、负载和轮数。不另加冒烟、重复轮、A/B、并发扫描或在线策略验证。不运行 CPU、mock、unittest 或 pytest，也不把它们写入通过线或交付。本 skill 不替代当前实现和 ClearTree README。

## 先读

1. `AGENTS.md` 的真实实验环境、设备、同步方式和隧道限制。设备、卡号、地址和隧道以该文件和用户当次指定为准。不把其中的地址、账户或口令写入脚本、日志或归档。
2. `python/sglang/srt/speculative/clear_tree_remote/README.md` 和这次行为的直接控制点。指标名以当前实现为准，不用历史字段。
3. 固定用户给出的角色、模型、设备、TP、page、注意力后端、静态内存、最大请求数、候选、容量、捕图 batch、bonus、ledger、lookahead、KV reuse、并发、条数、`max-new-tokens` 和轮数。

索引里最近一场已经跑完的目录只说明文件和脚本怎么放。当前可看 `test_sglang_liujg/clear_tree/validate/e2e/2026-11-18-c/`。不复制它的并发、条数、捕图 batch、开关或数字。历史 `raw/` 缺失时直接说明，不为取旧脚本重跑旧实验。

## 1. 建档

- 新建 `test_sglang_liujg/clear_tree/validate/e2e/<YYYY-MM-DD-后缀>/`，不覆盖既有场次。同时在 `test_sglang_liujg/clear_tree/validate/e2e/README.md` 的场次索引补一行：运行号、测什么、结果先写「未跑」。「测什么」写明条件和是 ABAB 还是同一条件的重复。跑完只改结果这一格，不把数字抄进索引。
- 开跑前写 `run.md` 的「事先」一节：范围、预热与 flush。每次试验用两位序号 `01`、`02`，负载、条件、第几次和阶段写在表内。看到结果后不回改这一节。跑完把数字和总结写进「结果」。`meta.json` 保留运行号和实验格；`voided` 只在你决定跳过这场汇总时设置。其余只记整场共用的开关，按这场实际因素增减。
- 在新场的 `tools/` 里写这次的启动脚本。模板的 `tools/` 是空目录，没有可直接执行的起步脚本。并发、条数、温度和 `max-new-tokens` 写在这场脚本里，并与试验表一致。图文负载的计时客户端用 `test_sglang_liujg/evaluation/eval_spectre_rsteller_serving.py` 或 `eval_spectre_watercaption_serving.py`；纯文本仍用 `eval_spectre_text_serving.py`。不回改其他场次的脚本。
- 每个试验使用和序号相同的新容器临时目录，日志放在 `raw/<序号>/`。失败记录、原始日志和脚本 hash 保留。重试是新场次，不把多次 client 输出拼成一轮。
- `raw/` 放双端日志、客户端 JSONL/CSV/manifest、包与数据 hash、同步前备份和清理证据。不创建 commit 或 branch，不改已有启动脚本和已批准计划。

## 2. 预检与窄同步

- 确认本机 Docker 和远端 SSH 可用。沙箱看不到 Docker socket 时，用相应权限在宿主机重试。
- 同步前和每次启动前确认：没有他人的非僵尸服务、HTTP 端口空闲、所选设备空闲、既有 `ssh -N` 的 PID、启动时间、UID、命令和 boot ID 不变。观测不完整或有冲突任务就停止，不结束他人任务。不新建、不杀掉 `ssh -N`，不运行 `sudo tc`。
- 本地和 NPU 是两份代码。先备份远端 ClearTree 包，再只同步本次授权包：NPU `/tmp` 中转、`docker cp`、容器内解包、属主改回。禁止直接写入无权限的包目录或覆盖整仓。
- 包摘要用 `.cursor/skills/clear-tree-e2e-validation/scripts/package_manifest.py`。本机和 NPU 容器各跑一次，再 diff 两份输出。需要把同一组文件打包时加 `--tar`。脚本排除 `__pycache__`、`.pyc`、`clear_tree_fit.json` 和 `.tmp`。核对数据身份。经 pty 抄回来的清单带 CR，diff 之前去掉。核对 tar 时只认拟合文件名；`clear_tree_fit.py` 是源码，不是拟合文件。
- 现行拟合有两份：greedy 的 `clear_tree_fit.json`，RPD 的 `clear_tree_fit.rpd.json`。打包只排除前者，所以 `.rpd.json`、`.bak.json`、`.bak2.json` 会进 tar。重新预拟合时，打包前把这些文件从 Draft 包目录移到本场 `raw/sync/`，记下 sha256，不删除。NPU 上的同名文件同样移走。Draft 才写现行拟合，NPU 上经常只有 bak；缺文件记 `ABSENT`，scp 找不到路径时不要让后续步骤退出。这些文件经常属主是 root，用挂载该目录的容器执行移动。
- 解包不会删掉目标目录里多出来的文件。复用拟合时，同步的保留集留下这两份现行文件。重新预拟合时，保留集不要包含它们，否则旧的 `.rpd.json` 会让 FIT 握手跳过 `need_prefit`。
- 口令辅助的 ssh 走 pty，调用方的标准输入不会送到远端。远端脚本放进 ssh 命令行，用 base64 传过去再解码执行，不用 `bash -s` 加本地 heredoc。`docker exec` 外面的 `< file` 重定向发生在宿主机；要读容器里的文件，把重定向放进 `bash -lc`。
- 启动前用 `PYTHONPATH=python python3 .cursor/skills/clear-tree-e2e-validation/scripts/parse_argv.py --role draft -- <Draft argv>`，Target 把 `--role` 换成 `target`。它调用 `prepare_server_args` 和 `apply_clear_tree_settings`，Draft 的 `on_npu` 为假，Target 为真，不启动服务。解析失败就不要启动。候选与该端包的默认列表相同时，省略 `--clear-tree-widths` 和 `--clear-tree-ks`。实验凭据不写入脚本或产物。

## 3. 运行

- 先建好该试验的容器临时目录，再把 stdout 重定向进去。目录不存在时重定向会让进程立刻退出。
- 先启动 Draft 并等到 HTTP 就绪，再启动 Target。Target 的内置预热会访问 Draft；Draft 未就绪时单独启动 Target 会形成 RPC 超时，这种启动不作为端到端结果。用户没有要求每轮冷启动时，一次拉起的服务可以连续跑该场的多个试验。要求冷启动时，每轮都重新走这一顺序。
- 要求重新预拟合时，`/health` 返回 200 还不够。发客户端前要看到：Draft 对将用的拟合路径打过 `no fit file`，Target 的 `prefit driver finished state=ready`，Draft 只写了一次本模式的拟合文件。捕图期间健康检查可以一直不是 200，这不是挂死。
- 启动参数核对角色、设备、TP、page、注意力后端、静态内存、最大请求数、候选、容量和捕图 batch。NPU 容器设置 `PYTHONPATH`。客户端在 NPU 上请求 Target HTTP。按试验表把图文或纯文本计时脚本及其同目录依赖拷到 NPU 的独立目录。输出文件名会带唯一 run id。用户没有要求 BLEU/ROUGE 时加 `--skip-metrics`。
- 正式客户端内部的一次预热和 flush 不计入吞吐。flush 是否成功看 Target 日志里的 `Cache flushed successfully` 和 HTTP 200。客户端在 flush 失败时仍会进入正式计时。`max-new-tokens` 是预算。
- 捕图和预热可以持续数十分钟。只根据日志是否继续推进、进程是否存活判断；静默本身不是挂死。用 `ps -p` 看进程。对容器内 uid 0 的进程执行 `kill -0` 会因权限失败，看起来像进程已经退出。设备、形状、捕图或正式请求失败时停止并留证。
- 性能用客户端自身计时。日志按完整行判断。可选模型导入提示，以及已证明为本场停止造成的停机输出，不是运行失败。

## 4. 验收

- 捕图证据是日志里的 `Capturing: bs=` 行。用户指定的 batch 都出现才算捕到。容量不够而没有出现的形状单独写出，不算捕图失败。捕获完成不等于运行时命中。
- 只记录当次矩阵打开的开关。Bonus 开启时，核对 Draft 的 `SGLANG_CLEAR_BONUS_PREDICTION`、lookahead、page size 和 KV reuse 是否与矩阵一致，并记录当前实现的 bonus 计数。周期计数是每个已刷窗口的增量，只加 Draft rank 0。窗口原因是 `idle` 表示该窗已刷出。没有指标窗口，或窗末尾没刷出的尾数，记为 missing，不记成 0。
- 指标名以当前 README 和实现为准。历史的 `opt_bonus_batch_*`、`skipped_graph`，以及 `is_bonus=False` 的普通层图复用，不当作批量 bonus 证据。客户端并发不等于每轮 batch。
- 日志收回 `raw/` 后运行 `python3 .cursor/skills/clear-tree-e2e-validation/scripts/accept_logs.py --raw <场次>/raw`。它按试验目录汇总 JSONL 的成功数、输出 token、正式 wall 和吞吐，以及总 `spec_accept_token_num / spec_draft_token_num`。Target 日志列出 `Capturing: bs=` 的 batch 与 ntpb、各 TP 的 `Cache flushed successfully` 和 `POST /flush_cache` 的 HTTP 200。Draft 日志加 rank 0 的 `[SR Draft round]`：`full`、`idle`、四个 `opt_bonus_*`；末窗不是 `idle` 时尾数记 `tail=missing`。正式 `d_round` 给出 `hold_ms`、`q_ms`、`hold_invalid` 和排除轮。脚本只打印数字，不判定通过。客户端产物名是 `client.<时间>-<id>.jsonl`。`accept_logs.py` 丢掉最后一次 `wrote fit file` 之前那一代 `d_round`。
- 正式阶段的 Target `t_verify` 从 Target 日志自己的 `prefit driver finished` 那一行之后算。不要用 Draft 或客户端的墙钟去切，NPU 时钟和 Draft 主机不是同一套。晚到 RPD 掩码在 `d_rpd`，按 `(session_id, rpc_seq)` 对打分轮。没对上的轮可以散在整段里，不只有进程结束时的尾数。
- 请求数、成功数、输出 token 和正式 wall time 要互相符合。吞吐是总输出 token 除以正式 wall。接受率是总 `spec_accept_token_num` 除以总 `spec_draft_token_num`，不用按请求平均的接受率。把上述数字写入 `run.md` 的「结果」、`summary.json` 和 `checks.json`，并保留 `meta.json` 和产物 hashes。客户端结束不等于验收通过。

## 5. 清理

- 启动时用 `python3 .cursor/skills/clear-tree-e2e-validation/scripts/process_identity.py <宿主机 PID>` 记录 NSpid、session、start ticks、UID 和 boot ID。发信号前再跑一次，这些字段一致才在容器内对 `container_nspid` 发信号。容器进程常常是 uid 0，宿主机用户发信号会被拒绝。只停止本场 owned session。本场 session 里还有 scheduler、detokenizer 和 inductor worker。sid 与 boot ID 一致后，对每个仍存活成员的 `container_nspid` 发信号。禁止 `pkill -f clear-tree-role`、按命令行广杀 Python，或停止 SSH 隧道。
- 列出进程和读 `/proc` 之间，成员可能已经退出。`missing /proc` 不是停止失败。这个循环不要用 `set -e`，否则后面的成员会被丢掉。
- 已经存在的交互式 `docker exec` 和 `tail` 不属于本场，不停止。清理证据要能看出 owned session 没有存活成员、HTTP 端口已释放、所选设备空闲、隧道身份未变。停完以后 session 里可以只剩僵尸，僵尸不是正在运行的服务。端口已释放是 `ss` 没有该端口的监听行；`awk` 退出码 0 不能当成端口仍被占用。
- 交付结果目录、实际命令、每轮请求与吞吐、图捕获、失败与重试、清理状态和未验证项。缺必要证据时写明阻塞。
