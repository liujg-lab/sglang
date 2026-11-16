# 验证耗时

对应文稿第 4 节。检查验证耗时随 K 和 KV 桶的变化。NPU 上用 Target 回传的 `device_us`；超出最大桶的轮次标为 eager，不和图内样本混在一次拟合里。

从 `../_template/` 复制到 `validate/cost-verify/<YYYY-MM-DD-a>/` 后再跑。
