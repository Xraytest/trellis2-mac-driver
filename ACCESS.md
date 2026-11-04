# applemini 稳定访问通路（2026-11-04 建成）

## 结论

```
本机 ──SSH(免密)──> ali01 (47.114.118.194:10022) ──frp──> applemini (127.0.0.1:22)
```

- **不再依赖** pi(192.168.1.3) → chat-lan → nc 跳板链（该链路会随机整机掉线、且 >1MB 输出静默截断）
- 支持 **scp/sftp 直传大文件**（21MB GLB 一次拉齐，无需 base64 分块）
- mac 侧 launchd 托管，开机自启 + 崩溃自动拉起（`KeepAlive`）

## 用法

```bash
# 命令
ssh -p 10022 yangmingchen@47.114.118.194 "hostname"

# 传文件（比跳板链的 gzip+400KB 分块快得多）
scp -P 10022 yangmingchen@47.114.118.194:~/trellis2-mac-driver/out/red_brick_fp32.glb ./
```

## 组成

| 位置 | 内容 |
|---|---|
| ali01 `/opt/frps/frps` | frp v0.71.1 服务端，`bindPort 7000`，token 认证，`allowPorts 10000-20000` |
| mac `~/frp/frp_0.71.1_darwin_arm64/frpc` | 客户端（在 mac 上直下 GitHub release） |
| mac `~/frp/frpc.toml` | `mac-ssh` → remotePort **10022**；`mac-trellis-http` → remotePort **10088** |
| mac `~/Library/LaunchAgents/com.underterra.frpc.plist` | launchd，`RunAtLoad` + `KeepAlive` |
| mac `~/.ssh/authorized_keys` | 已装本机 `id_ed25519.pub`（免密） |

## 运维

```bash
# 状态
ssh -p 10022 yangmingchen@47.114.118.194 "launchctl list | grep underterra; tail -5 ~/frp/frpc.launchd.log"

# 重启客户端
ssh -p 10022 ... "launchctl unload ~/Library/LaunchAgents/com.underterra.frpc.plist; \
                  launchctl load ~/Library/LaunchAgents/com.underterra.frpc.plist"

# 连通性自检（不连 SSH，只看端口）
nc -z 47.114.118.194 10022 && echo TUNNEL_UP
```

## 注意

- ali01 的 frps dashboard（7500）只绑 127.0.0.1，需 SSH 进 ali01 才能看；本次未使用
- `loginFailExit = false` + `heartbeatInterval 15/45`：网络抖动不会被判死，配合 launchd 自愈
- 若 ali01 重启，mac 侧 frpc 会自动重连，无需人工干预
