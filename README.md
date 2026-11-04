# trellis2-mac-driver

TRELLIS.2 shape 管线自构建驱动：绕开 ComfyUI 运行时的精度问题，在 applemini (M4/16GB)
上以可控精度（默认 fp32）直接用 torch 跑通「图 → GLB」，产出正确几何。

## 背景

2026-11-04 实测：ComfyUI 0.43 内置 TRELLIS.2 节点在 applemini 上连续 5 次
（int8 v1/v2、v3、v4、bf16 v5）产出碎片化错误几何；同 seed 同输入在 NVIDIA GPU
上几何正确。

### 根因（已二分定位，非精度问题）

逐阶段对拍（同样的 dump 在 mac 与 CUDA 上交叉解码）：

| 环节 | mac 结果 | 判定 |
|---|---|---|
| DINO 条件 tokens | norm 1026.38 vs CUDA 1026.47 | ✅ 健康 |
| 结构/shape DiT 采样 | bf16 与 fp32 产出**完全一致**（592,316 顶点同） | ✅ 健康（也排除精度假说） |
| shape 潜变量（拉到 CUDA 解码） | **1,185,700 面**（GPU 基线 1,189,776） | ✅ 健康 |
| **MPS 上的 shape VAE 解码** | 同一潜变量只解出 **208,642 面**（丢 82%） | ❌ **唯一损坏点** |

即：MPS 后端在 `FlexiDualGridVaeDecoder`（稀疏卷积 + `flexible_dual_grid_to_mesh`）
上算出错误结果，与 fp32/bf16 无关。**修复 = VAE 解码放到 CPU**（`--vae-device cpu`），
其余阶段仍走 MPS 加速。修复后 mac 产出 592,316 顶点 / 1,185,700 面，包围盒
1.0×0.299×0.432 对齐 GPU 基线 1.0×0.277×0.438。

## 组成

| 文件 | 作用 |
|---|---|
| `trellis2_mac_driver.py` | 自构建驱动：DINO 条件 → 结构 DiT 采样 → 结构 VAE 解码 → shape DiT 采样 → shape VAE 解码 → GLB |
| `crop_redbrick.py` | 经本机 ComfyUI(8188) 产 red_brick 裁切图（BiRefNet + ImageCropToMask，与 GPU 基准同配方） |
| `run_redbrick.sh` | 一键编排：ComfyUI 裁切 → 释放内存 → caffeinate + 驱动 fp32/MPS 全流程 |
| `audit_glb.py` | GLB 结构审计（verts/tris/bbox/退化面） |
| `ACCESS.md` | 稳定访问通路（frp 经 ali01 直连 mac:22） |

## 用法（applemini 上）

```bash
git clone https://github.com/Xraytest/trellis2-mac-driver.git
cd trellis2-mac-driver
bash run_redbrick.sh                  # fp32 全流程（VAE 自动走 CPU），产物 out/red_brick_fp32.glb
bash run_redbrick.sh --dtype bf16     # 对照实验用
```

`run_redbrick.sh` 的固定配方（16GB M5/M4 实测唯一可行且正确）：

```
--struct-dtype bf16 --struct-attention sub_quad   # 4096 token 稠密注意力，fp32 必 OOM
--dtype fp32 --attention split                    # shape 阶段分块注意力
--vae-device cpu                                  # ★ 绕开 MPS 解码缺陷（根因）
```

## 采样语义（逐条对照 comfy 0.43 源码钉死）

- `FLOW → CONST`：`x0 = x − σ·v`（v 参数化）；`calculate_input = identity`
- `ModelSamplingSD3(shift=5)`：`sigma(t)=time_snr_shift(5, t/1000)`，`timestep(σ)=σ·1000`
- `normal_scheduler`：`linspace(timestep(σ_max), timestep(σ_min), steps)` + 末尾补 0
- `prepare_noise`：`torch.manual_seed(seed)` CPU float32
- CFG + `RescaleCFG` flow 分支：x0 空间 per-item std 重标定，x0_final 取 cond item
- euler：`d = (x − denoised)/σ`；`x += d·(σ_next − σ)`
- 结构 latent `[1,32,16,16,16]`（模型取前 8 通道，输出 pad 回 32 通道）
- shape latent `[1,32,mt,1]`（padded 行输出 0，mask 重建）
- 结构解码：`decode_structure>0`（64³）→ `maxpool3d(2)>0.5` → argwhere coords
- shape 解码：`process_out(·std+mean)` → `decode_shape_slat(512)` → z_up→y_up `(x,z,−y)`

## 精度策略

- 采样算术恒 fp32；NN 权重精度可用 `--dtype` / 各阶段 `--*-dtype` 分别控制
- **`--vae-device cpu` 是本仓库的关键修复**：MPS 的 dense VAE 解码会给错网格
- `PYTORCH_ENABLE_MPS_FALLBACK=1`：MPS 未实现算子自动回落 CPU
- `--dump DIR`：落盘 cond/coords/struct/shape/mesh 中间量，跨设备/跨精度交叉解码对拍
  （本次根因就是这样定位的：把 mac 的 `shape.pt` 拿到 CUDA 上解码，直接证明潜变量没坏）
- `--struct-device/--shape-device/--vae-device/--dino-device`：单阶段换设备二分

## 已知限制

- 只覆盖 shape 管线（512 档，global cross-attn 路径）；texture 阶段未包含
- VAE 在 CPU 上解码约 13s（20 步采样合计 ~3 分钟整链）
- 结构阶段 fp32 + 4096 token 在 16GB MPS 必 OOM，必须 `--struct-dtype bf16`
