# trellis2-mac-driver

TRELLIS.2 shape 管线自构建驱动：绕开 ComfyUI 运行时的精度问题，在 applemini (M4/16GB)
上以可控精度（默认 fp32）直接用 torch 跑通「图 → GLB」，产出正确几何。

## 背景

2026-11-04 实测：ComfyUI 0.43 内置 TRELLIS.2 节点在 applemini 上连续 5 次
（int8 v1/v2、v3、v4、bf16 v5）产出碎片化错误几何；同 seed 同输入在 NVIDIA GPU
上几何正确。mac 日志证实 DiT 以 bf16 在 MPS 上执行——低精度数值是首要嫌疑。

## 组成

| 文件 | 作用 |
|---|---|
| `trellis2_mac_driver.py` | 自构建驱动：DINO 条件 → 结构 DiT 采样 → 结构 VAE 解码 → shape DiT 采样 → shape VAE 解码 → GLB |
| `crop_redbrick.py` | 经本机 ComfyUI(8188) 产 red_brick 裁切图（BiRefNet + ImageCropToMask，与 GPU 基准同配方） |
| `run_redbrick.sh` | 一键编排：ComfyUI 裁切 → 释放内存 → caffeinate + 驱动 fp32/MPS 全流程 |
| `audit_glb.py` | GLB 结构审计（verts/tris/bbox/连通性粗检） |

## 用法（applemini 上）

```bash
git clone https://github.com/Xraytest/trellis2-mac-driver.git
cd trellis2-mac-driver
bash run_redbrick.sh                  # fp32 全流程，产物 out/red_brick_fp32.glb
bash run_redbrick.sh --dtype bf16     # 复现原失败路径（对照实验用）
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

- 全部 NN 权重与前向 fp32（`--dtype fp32`）；采样算术恒 fp32
- `PYTORCH_ENABLE_MPS_FALLBACK=1`：MPS 未实现算子自动回落 CPU（正确性优先）
- `--dump DIR`：落盘 cond/coords/struct/shape/mesh 中间量，跨设备/跨精度对拍
- `--struct-device/--shape-device/--vae-device/--dino-device`：单阶段换设备二分根因
- `--dtype bf16` 可精确复现原失败路径，验证根因假设

## 已知限制

- 只覆盖 shape 管线（512 档，global cross-attn 路径）；texture 阶段未包含
- fp32 下 shape DiT 峰值约 6–7GB，16GB 统一内存可容纳；跑前先停 ComfyUI
