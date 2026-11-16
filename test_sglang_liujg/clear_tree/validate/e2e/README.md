# 端到端

对应文稿第 7 节。四条对照是 Target-only、固定 OPT、账本关闭的平表、两端账本打开的 ClearTree。主判据是按请求配对的接受长度，吞吐是次判据。

从 `../_template/` 复制到 `validate/e2e/<YYYY-MM-DD-a>/` 后再跑。同一场次内 A/B 交替，不把不同日期的吞吐并成一格。

`text_only_ledger.py` 是这条对照的客户端。计时沿用 `test_sglang_liujg/evaluation` 里的流式观察，默认 100 条、并发 4、温度 0、最多 1024 个新 token，不计算 BLEU / ROUGE。
