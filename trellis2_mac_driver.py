#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRELLIS.2 shape 管线自构建驱动（绕开 ComfyUI 运行时的 MPS 精度修复版）。

背景（2026-11-04）：applemini (M4/16GB, macOS 15.7, torch 2.9-dev) 上 ComfyUI 0.43 内置
TRELLIS.2 节点连续 5 次产出碎片化错误几何（int8 v1/v2、v3、v4、bf16 v5），同 seed 同输入
在 NVIDIA GPU (ComfyUI) 上唯一几何正确。mac 日志证实 DiT 以 bf16 在 MPS 上执行——
低精度数值是首要嫌疑。本驱动逐阶段直接用 torch 执行，精度可全程控制（默认 fp32），
支持 --dump 跨设备对拍与 per-stage 设备覆盖（二分根因）。

采样语义（逐条对照 comfy 0.43 源码钉死，非猜测）：
  * ModelType.FLOW -> CONST(model_sampling.py)：calculate_input=identity，
    calculate_denoised = x - sigma*model_output（v 参数化；model_base._apply_model 尾部调用）
  * ModelSamplingSD3(shift=5) -> ModelSamplingDiscreteFlow：sigma(t)=time_snr_shift(5,t/1000)，
    timestep(sigma)=sigma*1000（model_sampling.py:289-331）
  * normal_scheduler（samplers.py:671-693）：timesteps=linspace(timestep(sigma_max),
    timestep(sigma_min), steps)；末尾补 0
  * prepare_noise（sample.py:22-27）：generator=torch.manual_seed(seed)，CPU randn float32
  * 初始 x = CONST.noise_scaling(sigma_max=1, noise, 0) = noise
  * KSampler CFG：cond/uncond 同一 noise 复制成 batch 2（calc_cond_batch），
    cfg_function + RescaleCFG flow 分支（nodes_model_advanced.py）：
      x0_c/x0_u 为 apply_model 输出（已是 x0=x-σv）；x0_cfg = x0_u + cfg*(x0_c-x0_u)；
      per-item std(dims=1..) rescale；euler 用 x - x0_final 作方向
  * euler（k_diffusion/sampling.py:190-212）：d=(x-denoised)/sigma；x += d*(sigma_next-sigma)
  * 结构阶段 latent [1,32,16,16,16]（EmptyTrellis2LatentStructure），模型取前 8 通道
  * shape 阶段 latent [1,32,max_tokens,1]（Trellis2ShapeStage），padded 行模型输出 0
  * VaeDecodeStructureTrellis2：decode_structure>0（64³）→ maxpool3d(2)>0.5 → argwhere coords
  * VaeDecodeShapeTrellis：process_out(x*std+mean) → decode_shape_slat(res=512) →
    z_up 转 y_up：v=(x,z,-y)

用法：
  python3 trellis2_mac_driver.py --image crop.png --out out.glb --seed 12345 \
      [--device mps|cuda|cpu] [--dtype fp32|bf16|fp16] [--dump DIR]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# 必须在 import torch 前设置：MPS 缺算子时回落 CPU（正确性优先）
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT_ROOT = None      # --out 所在目录（允许写入的根）
DUMP_DIR = None      # --dump 目录（可为 None）
ARGS = None          # parse_args() 结果，main 里赋值


def log(msg):
    print(msg, flush=True)


def safe_out_path(raw, must_exist=False):
    """规范化 CLI 路径：禁止 .. 逃逸；解析为绝对路径；must_exist 时校验文件在。"""
    p = Path(raw)
    if ".." in p.parts:
        raise SystemExit("路径不允许包含 ..: %r" % raw)
    p = p.resolve()
    if must_exist and not p.is_file():
        raise SystemExit("文件不存在: %s" % p)
    return str(p)


def guarded_open(path, mode):
    """统一写入口：只允许写 OUT_ROOT / DUMP_DIR 白名单目录树内的文件。"""
    p = Path(path).resolve()
    roots = [r for r in (OUT_ROOT, DUMP_DIR) if r]
    if not roots:
        raise SystemExit("OUT_ROOT 未初始化，拒绝写入")
    if not any(str(p) == r or str(p).startswith(str(r) + os.sep) for r in roots):
        raise SystemExit("写入路径越出白名单目录: %s" % p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, mode)


def parse_args():
    global OUT_ROOT, DUMP_DIR, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="1024x1024 裁切图（ImageCropToMask 同款）")
    ap.add_argument("--out", required=True, help="输出 GLB 路径")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--struct-dtype", default=None, choices=["fp32", "bf16", "fp16"],
                    help="覆盖结构 DiT 精度（默认跟 --dtype）。结构阶段 4096 token 稠密注意力，"
                         "fp32 在 16GB MPS 会 OOM；bf16 为 ComfyUI 同档位")
    ap.add_argument("--shape-dtype", default=None, choices=["fp32", "bf16", "fp16"],
                    help="覆盖 shape DiT 精度（默认跟 --dtype）")
    ap.add_argument("--vae-dtype", default=None, choices=["fp32", "bf16", "fp16"],
                    help="覆盖 VAE 精度（默认跟 --dtype）")
    ap.add_argument("--dino-dtype", default=None, choices=["fp32", "bf16", "fp16"],
                    help="覆盖 DINO 精度（默认跟 --dtype）")
    ap.add_argument("--attention", default="split",
                    choices=["split", "sub_quad", "pytorch", "default"],
                    help="注意力实现。split=分块（fp32 友好）；pytorch=SDPA（MPS 4096token OOM）")
    ap.add_argument("--struct-attention", default=None,
                    choices=["split", "sub_quad", "pytorch"],
                    help="仅结构阶段覆盖注意力实现（该阶段 4096 token 是内存大头）")
    ap.add_argument("--comfy", default=os.path.expanduser("~/ComfyUI"))
    ap.add_argument("--steps-s", type=int, default=12, help="结构阶段步数")
    ap.add_argument("--steps-h", type=int, default=20, help="shape 阶段步数")
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--rescale", type=float, default=0.7)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--dump", default=None, help="中间量 dump 目录（跨设备对拍）")
    ap.add_argument("--struct-device", default=None, help="覆盖结构 DiT 设备（bisect 用）")
    ap.add_argument("--shape-device", default=None, help="覆盖 shape DiT 设备")
    ap.add_argument("--vae-device", default=None, help="覆盖 VAE 设备")
    ap.add_argument("--dino-device", default=None, help="覆盖 DINO 设备")
    args = ap.parse_args()
    args.image = safe_out_path(args.image, must_exist=True)
    args.out = safe_out_path(args.out)
    args.dump = safe_out_path(args.dump) if args.dump else None
    OUT_ROOT = str(Path(args.out).parent.resolve())
    DUMP_DIR = args.dump
    ARGS = args
    return args


# ---------------------------------------------------------------- sigma 调度
def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1.0 + (alpha - 1.0) * t)


class FlowSampling:
    """ModelSamplingDiscreteFlow(shift, multiplier=1000) 的最小复刻（fp32）。"""

    def __init__(self, shift):
        self.shift = shift
        self.multiplier = 1000.0
        ts = self.sigma((torch.arange(1, 1001, dtype=torch.float32) / 1000.0) * self.multiplier)
        self.sigmas = ts

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)


def normal_scheduler(ms, steps):
    """comfy/samplers.py normal_scheduler（sgm=False 分支）逐行复刻。"""
    start = ms.timestep(ms.sigma_max)
    end = ms.timestep(ms.sigma_min)
    append_zero = True
    if math.isclose(float(ms.sigma(end)), 0.0, abs_tol=0.00001):
        steps += 1
        append_zero = False
    timesteps = torch.linspace(start, end, steps, dtype=torch.float32)
    sigs = [float(ms.sigma(ts)) for ts in timesteps]
    if append_zero:
        sigs += [0.0]
    return torch.FloatTensor(sigs)


def prepare_noise(shape, seed):
    """comfy/sample.py prepare_noise：CPU generator manual_seed，float32。"""
    generator = torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, layout=torch.strided,
                       generator=generator, device="cpu")


def load_prefixed(path, prefix, dtype):
    """safe_open 惰性读取指定前缀张量（不整文件载入，控内存）。"""
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                out[k[len(prefix):]] = f.get_tensor(k).to(dtype)
    if not out:
        raise RuntimeError("前缀 %r 在 %s 中没有张量" % (prefix, path))
    return out


def free_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cpu":
        pass


def force_attention_mode(mode, t2m=None):
    """强制 comfy 注意力实现。t2m 不为 None 时同时改 trellis2.model 的调用点绑定
    （model.py 在 import 时按值把 optimized_attention 绑进自己命名空间，运行时改
    t2m.optimized_attention 即可生效，因此支持 per-stage 切换）。

    MPS 实测（2026-11-04, 16GB）：
      * SDPA（pytorch）对 4096-token 稠密注意力回落 math 路径，整块 score
        矩阵（bf16 [2,12,4096,4096] = 768MB/层）直接实例化 → OOM
      * sub_quad 分块（ComfyUI 默认）bf16 可跑，但 fp32 仍爆
      * split 分块对 fp32 最友好（q 分块逐段算 scores，瞬态 ~200MB 级）
    """
    import comfy.ldm.modules.attention as comfy_attn
    fn = {"split": comfy_attn.attention_split,
          "sub_quad": comfy_attn.attention_sub_quad,
          "pytorch": comfy_attn.attention_pytorch}.get(mode)
    if fn is None:
        log("  attention: 保持 comfy 默认选择")
        return fn
    comfy_attn.optimized_attention = fn
    comfy_attn.optimized_attention_masked = fn
    if t2m is not None:
        t2m.optimized_attention = fn
        t2m.optimized_attention_masked = fn
    log("  attention: forced %s" % mode)
    return fn


# ---------------------------------------------------------------- DINO 条件
def encode_dino(image_path, comfy_dir, ckpt_path, device, dt, ops):
    """nodes_trellis2._dinov3_encode(512) 语义：lanczos 512 + normalize +
    skip_norm_elementwise 前向 -> [1,1029,1024] tokens。"""
    import comfy.utils
    import comfy.image_encoders.dino3 as dino3

    cfg_path = os.path.join(comfy_dir, "comfy", "image_encoders", "dino3_large.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    model = dino3.DINOv3ViTModel(cfg, dtype=dt, device=device, operations=ops)
    sd = comfy.utils.load_torch_file(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    critical = [k for k in missing if not k.endswith("mask_token")]
    if critical:
        raise RuntimeError("DINO 缺失权重: %s" % critical[:8])
    if unexpected:
        log("  dino unexpected keys: %d（忽略）" % len(unexpected))
    model.eval()

    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    arr = torch.from_numpy(np.asarray(img)).to(torch.float32) / 255.0   # [H,W,3]
    img_bchw = arr.permute(2, 0, 1).unsqueeze(0).to(device)             # [1,3,H,W]
    img512 = comfy.utils.common_upscale(img_bchw, 512, 512, "lanczos", "disabled")
    mean = torch.tensor(cfg.get("image_mean", [0.485, 0.456, 0.406]), device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("image_std", [0.229, 0.224, 0.225]), device=device).view(1, 3, 1, 1)
    img512 = (img512 - mean) / std
    with torch.no_grad():
        tokens = model(img512.to(dt), skip_norm_elementwise=True)[0]
    tokens = tokens.float()
    log("  dino tokens: %s" % (tuple(tokens.shape),))
    del model, sd
    free_cache(device)
    return tokens


# ---------------------------------------------------------------- 采样核心
def flow_euler_step(x, sigma, sigma_next, x0_c, x0_u, cfg_scale, rescale):
    """cfg_function(RescaleCFG flow 分支) + euler，全部 fp32，per-item std。

    x: [B,...]（B=2: cond/uncond 同 noise）；x0_*: apply_model 输出（已是 x0）。
    """
    x0_cfg = x0_u + cfg_scale * (x0_c - x0_u)
    dims = tuple(range(1, x0_c.ndim))
    ro_pos = x0_c.std(dim=dims, keepdim=True)
    ro_cfg = x0_cfg.std(dim=dims, keepdim=True).clamp(min=1e-8)
    x0_rescaled = x0_cfg * (ro_pos / ro_cfg)
    x0_final = rescale * x0_rescaled + (1.0 - rescale) * x0_cfg
    x0_final = x0_final[:1]
    d = (x[:1] - x0_final) / sigma
    return x[:1] + d * (sigma_next - sigma)


def sample_stage(model_fn, x, sigmas, cond, zeros_cond, tag, cfg_scale, rescale, multiplier,
                 step_cache_device=None):
    """通用采样循环。model_fn(x2, t_vec, cond2) -> v [B,...]（v 参数化）。

    x: [1,...] 初始噪声。返回最终 x（latent 空间）。
    step_cache_device 不为 None 时每步后清该设备缓存（MPS 显存碎片控制）。
    """
    steps = len(sigmas) - 1
    for i in range(steps):
        sig = float(sigmas[i])
        sig_next = float(sigmas[i + 1])
        t_vec = torch.full((2,), sig * multiplier, dtype=torch.float32, device=cond.device)
        x_in = torch.cat([x, x], dim=0)
        t0 = time.time()
        with torch.inference_mode():
            v_c, v_u = model_fn(x_in, t_vec, torch.cat([cond, zeros_cond], dim=0))
        v_c = v_c.float()
        v_u = v_u.float()
        x0_c = x_in[:1] - v_c * sig
        x0_u = x_in[1:] - v_u * sig
        x = flow_euler_step(x, sig, sig_next, x0_c, x0_u, cfg_scale, rescale)
        if step_cache_device is not None:
            free_cache(step_cache_device)
        log("  %s step %2d/%d sigma=%.5f x_std=%.4f (%.1fs)"
            % (tag, i + 1, steps, sig, float(x.float().std()), time.time() - t0))
    return x


# ---------------------------------------------------------------- GLB 导出
def export_glb(out_path, vertices, triangles, open_fn):
    """最小 GLB 写出（单 mesh primitive，uint32 索引）。open_fn 是 guarded_open。"""
    vertices = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 3)
    tri = np.ascontiguousarray(triangles, dtype=np.uint32).reshape(-1, 3)
    M, T = vertices.shape[0], tri.shape[0]

    buffer_views, accessors, bin_parts = [], [], []
    off = 0

    def add_view(data_bytes, target):
        nonlocal off
        buffer_views.append({"buffer": 0, "byteOffset": off,
                             "byteLength": len(data_bytes), "target": target})
        off += len(data_bytes)
        bin_parts.append(data_bytes)
        return len(buffer_views) - 1

    pos_bv = add_view(vertices.reshape(-1).tobytes(), 34962)
    pos_min = [float(vertices[:, i].min()) for i in range(3)] if M else [0.0] * 3
    pos_max = [float(vertices[:, i].max()) for i in range(3)] if M else [0.0] * 3
    accessors.append({"bufferView": pos_bv, "componentType": 5126, "count": M,
                      "type": "VEC3", "min": pos_min, "max": pos_max})
    idx_acc = None
    if T > 0:
        idx_bv = add_view(tri.reshape(-1).tobytes(), 34963)
        accessors.append({"bufferView": idx_bv, "componentType": 5125,
                          "count": T * 3, "type": "SCALAR"})
        idx_acc = len(accessors) - 1
    while off % 4:
        add_view(b"\x00", 0)

    mesh = {"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}
    if idx_acc is not None:
        mesh["primitives"][0]["indices"] = idx_acc
    gltf = {
        "asset": {"version": "2.0", "generator": "trellis2-mac-driver"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "trellis2_shape"}],
        "meshes": [mesh],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": off}],
    }
    glb = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad = (4 - len(glb) % 4) % 4
    glb += b" " * pad
    bin_chunk = b"".join(bin_parts)
    bpad = (4 - len(bin_chunk) % 4) % 4
    bin_chunk += b"\x00" * bpad
    total = 12 + 8 + len(glb) + 8 + len(bin_chunk)
    with open_fn(out_path, "wb") as f:
        f.write(np.asarray([0x46546C67, 2, total], "<I").tobytes())
        f.write(np.asarray([len(glb), 0x4E4F534A], "<I").tobytes())
        f.write(glb)
        f.write(np.asarray([len(bin_chunk), 0x004E4942], "<I").tobytes())
        f.write(bin_chunk)
    return total


# ---------------------------------------------------------------- 主流程
def main():
    args = parse_args()

    def stage_dt(name):
        raw = getattr(args, name) or args.dtype
        return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[raw]

    DT_S = stage_dt("struct_dtype")    # 结构 DiT
    DT_H = stage_dt("shape_dtype")     # shape DiT
    DT_V = stage_dt("vae_dtype")       # VAE
    DT_D = stage_dt("dino_dtype")      # DINO
    dev = torch.device(args.device)
    DEV_STRUCT = torch.device(args.struct_device or args.device)
    DEV_SHAPE = torch.device(args.shape_device or args.device)
    DEV_VAE = torch.device(args.vae_device or args.device)
    DEV_DINO = torch.device(args.dino_device or args.device)

    comfy_dir = os.path.abspath(os.path.expanduser(args.comfy))
    if comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)

    import comfy.ops  # noqa: E402
    import comfy.utils  # noqa: E402
    import comfy.latent_formats  # noqa: E402
    import comfy.ldm.trellis2.model as t2m  # noqa: E402
    import comfy.ldm.trellis2.vae as t2v  # noqa: E402

    attn_fn = force_attention_mode(args.attention, t2m)
    struct_attn_fn = None
    if args.struct_attention and args.struct_attention != args.attention:
        import comfy.ldm.modules.attention as comfy_attn
        struct_attn_fn = {"split": comfy_attn.attention_split,
                          "sub_quad": comfy_attn.attention_sub_quad,
                          "pytorch": comfy_attn.attention_pytorch}[args.struct_attention]

    ops = comfy.ops.disable_weight_init  # fp32 权重 + fp32 计算 -> 纯 torch 模块
    model_bf16 = os.path.join(comfy_dir, "models", "diffusion_models", "trellis_2_bf16.safetensors")
    shape_vae_path = os.path.join(comfy_dir, "models", "vae", "trellis_2_shape_vae_bf16.safetensors")
    dino_ckpt = os.path.join(comfy_dir, "models", "clip_vision", "dino_v3_vit_l.safetensors")
    shape_slat_fmt = comfy.latent_formats.Trellis2ShapeSLAT()

    cfg_scale = args.cfg
    rescale = args.rescale
    shift = args.shift
    multiplier = 1000.0

    log("== TRELLIS.2 自构建驱动 ==")
    log("device=%s dtype=%s struct=%s shape=%s vae=%s dino=%s seed=%d steps=%d/%d "
        "cfg=%.2f rescale=%.2f shift=%.1f"
        % (args.device, args.dtype, DT_S, DT_H, DT_V, DT_D, args.seed,
           args.steps_s, args.steps_h, cfg_scale, rescale, shift))

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    def dump_save(obj, name):
        if args.dump:
            with guarded_open(os.path.join(args.dump, name), "wb") as f:
                torch.save(obj, f)

    # ---- [1/6] DINO 条件
    t0 = time.time()
    cond = encode_dino(args.image, comfy_dir, dino_ckpt, DEV_DINO, DT_D, ops).to(dev)
    zeros_cond = torch.zeros_like(cond)
    log("[1/6] dino cond %.1fs" % (time.time() - t0))
    dump_save({"cond_tokens": cond.cpu()}, "cond.pt")

    # ---- [2/6] 结构 DiT 加载
    t0 = time.time()
    log("[2/6] structure DiT 加载…")
    sd = load_prefixed(model_bf16, "model.structure_model.", DT_S)
    struct_model = t2m.SparseStructureFlowModel(
        resolution=16, in_channels=8, out_channels=8, model_channels=1536,
        cond_channels=1024, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        operations=ops, device=DEV_STRUCT, dtype=DT_S,
    )
    struct_model.load_state_dict(sd, strict=True)
    struct_model.eval().to(DEV_STRUCT)
    del sd
    free_cache(DEV_STRUCT)

    ms = FlowSampling(shift)
    sigmas_s = normal_scheduler(ms, args.steps_s)
    noise_s = prepare_noise((1, 32, 16, 16, 16), args.seed).to(dev)
    cond_s = cond.to(DEV_STRUCT)
    zeros_s = torch.zeros_like(cond_s)

    def struct_fn(x_in, t_vec, c2):
        x8 = x_in[:, :8].to(DT_S)
        out = struct_model(x8, t_vec.to(DT_S), c2.to(DT_S))
        # Trellis2 wrapper：8ch 输出 pad 回 32ch（sampler 在 32ch 空间做 euler）
        return torch.nn.functional.pad(out.float(), (0, 0, 0, 0, 0, 0, 0, 24))

    # ---- [3/6] 结构采样
    if struct_attn_fn is not None:
        t2m.optimized_attention = struct_attn_fn
        t2m.optimized_attention_masked = struct_attn_fn
        log("  struct attention: %s" % args.struct_attention)
    log("[3/6] structure 采样…")
    x_struct = sample_stage(struct_fn, noise_s, sigmas_s, cond_s, zeros_s, "struct",
                            cfg_scale, rescale, multiplier, step_cache_device=DEV_STRUCT)
    if struct_attn_fn is not None:
        t2m.optimized_attention = attn_fn
        t2m.optimized_attention_masked = attn_fn
    log("[3/6] structure 完成 %.1fs latent_std=%.4f"
        % (time.time() - t0, float(x_struct.float().std())))
    dump_save({"x_struct": x_struct[:1, :8].float().cpu(), "sigmas": sigmas_s}, "struct.pt")
    del struct_model, cond_s, zeros_s
    free_cache(DEV_STRUCT)
    cond = cond.to(dev)

    # ---- [4/6] 结构 VAE 解码 -> coords
    t0 = time.time()
    log("[4/6] 结构 VAE 解码…")
    vae_sd = comfy.utils.load_torch_file(shape_vae_path)
    shape_vae = t2v.ShapeVae().to(dtype=DT_V)
    shape_vae.load_state_dict(vae_sd, strict=True)
    shape_vae.eval().to(DEV_VAE)
    del vae_sd
    free_cache(DEV_VAE)

    x8 = x_struct[:1, :8].to(DEV_VAE).to(DT_V)
    with torch.no_grad():
        decoded = shape_vae.decode_structure(x8) > 0              # [1,1,64,64,64]
    decoded = torch.nn.functional.max_pool3d(decoded.float(), 2, 2, 0) > 0.5  # -> 32³
    # 对齐 ComfyUI：VaeDecodeStructureTrellis2 产 voxel.data [1,32,32,32]，
    # Trellis2ShapeStage 在 argwhere 前 unsqueeze(1) 回 [1,1,32,32,32]（5D），
    # 再取 [:, [0,2,3,4]] = (batch, x, y, z)。漏掉 unsqueeze 会索引错列。
    voxel = decoded.squeeze(1).float()                            # [1,32,32,32]
    coords = torch.argwhere(voxel.unsqueeze(1).bool())[:, [0, 2, 3, 4]].int()
    n_coords = coords.shape[0]
    log("[4/6] 结构解码 %.1fs occupancy=%d voxels (32³)" % (time.time() - t0, n_coords))
    dump_save({"coords": coords.cpu(), "voxel": voxel.cpu()}, "coords.pt")
    if n_coords == 0:
        raise SystemExit("FATAL: 结构潜变量解码出空体素——结构阶段已坏")
    del x_struct, x8, decoded
    free_cache(DEV_VAE)

    # ---- [5/6] shape DiT 采样
    t0 = time.time()
    log("[5/6] shape DiT (img2shape_512) 加载…")
    sd = load_prefixed(model_bf16, "model.img2shape_512.", DT_H)
    slat_model = t2m.SLatFlowModel(
        resolution=32, in_channels=32, out_channels=32, model_channels=1536,
        cond_channels=1024, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        image_attn_mode="global",
        operations=ops, device=DEV_SHAPE, dtype=DT_H,
    )
    slat_model.load_state_dict(sd, strict=True)
    slat_model.eval().to(DEV_SHAPE)
    del sd
    free_cache(DEV_SHAPE)

    coords_d = coords.to(DEV_SHAPE)
    counts = torch.bincount(coords_d[:, 0].long(), minlength=1)
    max_tokens = int(counts.max().item())
    noise_h = prepare_noise((1, 32, max_tokens, 1), args.seed).to(dev)
    cond_h = cond.to(DEV_SHAPE)
    zeros_h = torch.zeros_like(cond_h)
    mask = (torch.arange(max_tokens, device=DEV_SHAPE).unsqueeze(0)
            < counts.to(DEV_SHAPE).unsqueeze(1))

    def shape_fn(x_in, t_vec, c2):
        xb = x_in.squeeze(-1).transpose(1, 2).to(DT_H)      # [2, mt, 32]
        feats_flat = xb[mask.expand(2, -1)]                    # [2N, 32]
        coords2 = torch.cat([coords_d, coords_d.clone()], dim=0)
        coords2[coords_d.shape[0]:, 0] += 1
        x_st = t2v.SparseTensor(feats=feats_flat, coords=coords2,
                                shape=torch.Size([2, 32]))
        out = slat_model(x_st, t_vec.to(DT_H), c2.to(DT_H))
        padded = torch.zeros((2, max_tokens, out.feats.shape[-1]),
                             device=DEV_SHAPE, dtype=out.feats.dtype)
        padded[mask.expand(2, -1)] = out.feats
        return padded.transpose(1, 2).unsqueeze(-1).float()     # [2, 32, mt, 1]

    log("[5/6] shape 采样…")
    x_shape = sample_stage(shape_fn, noise_h, normal_scheduler(ms, args.steps_h),
                           cond_h, zeros_h, "shape", cfg_scale, rescale, multiplier,
                           step_cache_device=DEV_SHAPE)
    log("[5/6] shape 完成 %.1fs" % (time.time() - t0))

    # 有效 token 的 latent（latent 空间原始值）
    xb = x_shape[:1].squeeze(-1).transpose(1, 2)[0].float()     # [mt, 32]
    slat_latent = xb[mask[0]]                                   # [N, 32]
    dump_save({"x_shape": slat_latent.cpu(), "n": n_coords}, "shape.pt")
    del slat_model, cond_h, zeros_h, noise_h
    free_cache(DEV_SHAPE)
    cond = None
    free_cache(dev)

    # ---- [6/6] shape VAE 解码 -> mesh -> GLB
    t0 = time.time()
    log("[6/6] shape VAE 解码 + GLB…")
    slat_feats = shape_slat_fmt.process_out(slat_latent.to(DEV_VAE))  # *std+mean
    coords_v = coords.to(DEV_VAE).clone()
    coords_v[:, 0] = 0
    st = t2v.SparseTensor(feats=slat_feats.to(DT_V), coords=coords_v,
                          shape=torch.Size([1, 32]))
    with torch.no_grad():
        mesh, subs = shape_vae.decode_shape_slat(st, 512)
    verts, faces = mesh[0]
    verts = torch.stack([verts[..., 0], verts[..., 2], -verts[..., 1]],
                        dim=-1).float().cpu()                   # z_up->y_up
    faces = faces.int().cpu()
    log("[6/6] 解码 %.1fs: %d verts / %d tris"
        % (time.time() - t0, verts.shape[0], faces.shape[0]))

    size = export_glb(args.out, verts.numpy(), faces.numpy(), guarded_open)
    log("GLB 写出: %s (%d bytes)" % (args.out, size))

    if args.dump:
        import hashlib
        h = hashlib.sha256()
        with guarded_open(args.out, "rb") as f:
            h.update(f.read())
        dump_save({"verts": verts, "faces": faces, "glb_sha256": h.hexdigest()}, "mesh.pt")
        log("dump sha256(glb)=%s" % h.hexdigest())
    log("DONE")


if __name__ == "__main__":
    main()
