# 在线实验

一次运行一个目录，运行号是 `validate/<实验>/<YYYY-MM-DD-a>/`。从 `_template/` 复制 `run.md`、`meta.json`、`raw/` 和 `tools/`，再开测。各实验格的说明在各自的 README。已经归档的场次保持当时的文件。

在该场 `tools/` 里写这次的启动脚本。日志和客户端输出放在 `raw/01/` 这样的子目录。

`_template/meta.json` 是起稿，不是每次都要填满的字段表。复制后保留 `run_id`、`experiment`、`voided`、`void_reason`。其他键按这场实际固定的因素增减。整场共用的开关放这里，各次试验的差异写在 `run.md`。不写地址、口令和模型路径。`run.md` 和 `meta.json` 可以提交，`raw/` 不进仓库。

开跑前填写 `run.md` 的「事先」。跑完把数字和总结写入「结果」。`voided` 只在你决定跳过这场汇总时设为 true。
