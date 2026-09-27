# STANDALONE_REMOTE 跨机隧道

Target 跑在 `10.20.25.3`，Draft 跑在 `172.22.14.164`。两边不能直连，ZMQ RPC 经跳板机 `10.20.25.6` 转发。本文只说明怎么接，不代替实际启动。

## 机器

| 角色 | 主机名 | 地址 | 能做什么 |
| --- | --- | --- | --- |
| Draft | `164` | `172.22.14.164` | 只能 SSH 到 `10.42.206.123` |
| 跳板 | `bms-82820586-006` | 对外 `10.42.206.123`，内网 `10.20.25.6` | 能 SSH 到 `10.20.25.3`，不能访问 164 |
| Target | `bms-82820586-003` | `10.20.25.3` | 只能 SSH 到内网 `10.20.25.6`，不能访问 `10.42.206.123` 和 164 |

`10.42.206.123` 和 `10.20.25.6` 是同一台跳板机。164 使用外网地址，`10.20.25.3` 使用内网地址。

已确认的限制：

- `10.20.25.3` 的 sshd 拒绝远程转发（`ssh -R`），允许本地转发（`ssh -L`）。
- 跳板机允许 `ssh -R`，但只绑 `127.0.0.1`。写成 `10.20.25.6:30019` 时，实际仍落在 `127.0.0.1:30019`。
- 两台 BMS 上的 `liujg` 没有 sudo，不能改 `sshd` 或做 iptables。
- `10.20.25.3` 禁止使用 crontab。跳板机没有为 `liujg` 打开 linger，用户级 systemd 在注销后不会保留。

## 协议

Draft 绑定 ZMQ ROUTER，Target 作为 DEALER 去连接。控制和视觉数据走同一条 TCP，端口是 `30019`。HTTP 只打 Target。

`python/sglang/srt/speculative/standalone_remote/sr_transport.py` 里，`--standalone-remote-addr` 的字符串若是 `127.0.0.1` 或 `0.0.0.0`，两端都改用本机 IPC，不会进入 TCP 隧道。

- Target 写 `localhost`，连接 `tcp://localhost:30019`，进入本机的 `ssh -L`。
- Draft 写 `172.22.14.164`，实际绑定 `tcp://*:30019`。隧道再连它的 `127.0.0.1:30019`。

```mermaid
flowchart LR
  http["HTTP 请求"] --> target["Target 10.20.25.3"]
  target -->|"tcp://localhost:30019"| localFwd["003 上的 ssh -L"]
  localFwd --> jump["006 的 127.0.0.1:30019"]
  jump -->|"164 上的 ssh -R"| draft["Draft 164 的 tcp://*:30019"]
```

## 隧道命令

两条命令分属两台机器，都不要在跳板机 `bms-82820586-006` 上执行。`ssh -L` 的监听开在执行命令的那台机器上。006 上若已有 164 建好的 `ssh -R`，`127.0.0.1:30019` 已被占用；再在 006 上执行下面的 `-L` 会报：

```text
bind [127.0.0.1]:30019: Address already in use
channel_setup_fwd_listener_tcpip: cannot listen to port: 30019
Could not request local forwarding.
```

即使停掉现有监听后在 006 上把 `-L` 跑成功，端口仍在跳板机上，`10.20.25.3` 的 Target 还是连不到。

`~/.ssh/tun3008` 尚未生成。密钥写入跳板机 `authorized_keys` 之前，不要加 `-i`。找不到该文件时 SSH 会提示 `Identity file ... not accessible`，然后回退到密码。

在 `172.22.14.164` 上保持这条反向转发。它在跳板机上打开 `127.0.0.1:30019`，有连接时再由 164 连回本机 `127.0.0.1:30019`：

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:30019:127.0.0.1:30019 \
  liujg@10.42.206.123
```

在 `bms-82820586-003`（`10.20.25.3`）上保持这条本地转发：

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:30019:127.0.0.1:30019 \
  liujg@10.20.25.6
```

成功后命令停住且没有输出。006 上已有的 `30019` 监听要保留，那是 164 的入口。

密码不要写进脚本。以后若在 164 和 `10.20.25.3` 各生成一把专用密钥，只把公钥追加到跳板机的 `authorized_keys`，再给上面两条命令加上 `-i ~/.ssh/tun3008`。

## 启动顺序

1. 164 上的 `ssh -R` 和 `10.20.25.3` 上的 `ssh -L` 都已保持运行。
2. 在 164 启动 Draft：

```bash
--speculative-algorithm STANDALONE_REMOTE \
--standalone-remote-role draft \
--standalone-remote-addr 172.22.14.164 \
--standalone-remote-port 30019
```

3. 在 `10.20.25.3` 启动 Target：

```bash
--speculative-algorithm STANDALONE_REMOTE \
--standalone-remote-role target \
--standalone-remote-addr localhost \
--standalone-remote-port 30019 \
--port 30000
```

4. HTTP 请求发到 `10.20.25.3:30000`。从 164 访问这个 HTTP 端口仍然不通。

两端词表必须一致。默认 RPC 超时是 5000ms（`--standalone-remote-rpc-timeout-ms`）。
