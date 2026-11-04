#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRELLIS.2 texture（贴图+材质）阶段：tex 采样 → tex VAE 解码 → UV 展开 → 烘焙 → 带 PBR 材质 GLB。

语义逐条对照 comfy 0.43 源码：
  * Trellis2TextureStage：latent = zeros[1,32,max_tokens,1]，extras 带
    trellis2_shape_slat = shape_latent["samples"]（latent 空间原始值）
  * model.py texture_generation 分支：x_st.feats = cat([x, slat], -1)（32+32=64 通道）
    再过 input_layer(64→1536)；uncond 分支同样保留 slat，只把 cond 换零
  * VaeDecodeTextureTrellis：
      feats = Trellis2TexSLAT.process_out(samples)      # latent*std + mean
      voxel = TextureVae.decode_tex_slat(feats, shape_subdivides)
            = txt_dec(slat, guide_subs=subs) * 0.5 + 0.5
      6 通道：base_color(0:3) / metallic(3) / roughness(4) / alpha(5)，均在 [0,1]
      tex_resolution = coord_resolution * 16
      coords z_up → y_up：(x, y, z) → (x, z, R-1-y)，R = tex_resolution
  * 后处理（复用 ComfyUI 官方节点，保证与 GPU 基线同一条代码路径）：
      UnwrapMesh → BakeTextureFromVoxel → ApplyTextureToMesh → mesh_item_to_glb_bytes
"""
import torch


def _log(*parts):
    """日志：逗号拼接，避免任何字符串格式化写法。"""
    print(*parts, flush=True)


def tex_sample(model, x, sigmas, cond, zeros_cond, multiplier=1000.0):
    """texture DiT 采样（v 参数化 + CFG + RescaleCFG，与 shape 阶段同一套语义）。

    x: [1,32,mt,1] 初始噪声；model: 包装函数，签名 fn(x_dup2, t_vec, cond2) -> v。
    返回 latent 空间的 tex latent [1,32,mt,1]。
    """
    steps = len(sigmas) - 1
    for i in range(steps):
        sig = float(sigmas[i])
        sig_next = float(sigmas[i + 1])
        t_vec = torch.full((2,), sig * multiplier, dtype=torch.float32, device=cond.device)
        x_in = torch.cat([x, x], dim=0)
        with torch.inference_mode():
            v_c, v_u = model(x_in, t_vec, torch.cat([cond, zeros_cond], dim=0))
        v_c = v_c.float()
        v_u = v_u.float()
        x0_c = x_in[:1] - v_c * sig
        x0_u = x_in[1:] - v_u * sig
        # cfg_function + RescaleCFG(flow)：per-item std 重标定
        x0_cfg = x0_u + 7.5 * (x0_c - x0_u)
        dims = tuple(range(1, x0_c.ndim))
        ro_pos = x0_c.std(dim=dims, keepdim=True)
        ro_cfg = x0_cfg.std(dim=dims, keepdim=True).clamp(min=1e-8)
        x0_final = 0.7 * (x0_cfg * (ro_pos / ro_cfg)) + 0.3 * x0_cfg
        x0_final = x0_final[:1]
        d = (x[:1] - x0_final) / sig
        x = x[:1] + d * (sig_next - sig)
        _log("  tex  step", i + 1, "/", steps, "sigma=", round(sig, 5),
             "x_std=", round(float(x.float().std()), 4))
    return x


def decode_tex(tex_vae, tex_latent, coords, subs, coord_resolution, dt, device):
    """VaeDecodeTextureTrellis 的等价实现。返回 (vox_coords[N,3] y_up, colors[N,6], tex_res)。"""
    import comfy.ldm.trellis2.vae as t2v
    import comfy.latent_formats

    fmt = comfy.latent_formats.Trellis2TexSLAT()
    feats = fmt.process_out(tex_latent.to(device).float())          # latent*std + mean
    c = coords.to(device).clone()
    c[:, 0] = 0
    st = t2v.SparseTensor(feats=feats.to(dt), coords=c, shape=torch.Size([1, 32]))
    guide = [s.to(device) for s in subs]
    with torch.inference_mode():
        voxel = tex_vae.decode_tex_slat(st.to(dt), guide)
    color = voxel.feats.float().cpu()
    vc = voxel.coords.cpu()

    tex_res = int(coord_resolution) * 16
    R = tex_res
    if vc.shape[-1] == 4:
        batch_col = vc[:, :1]
        sp = vc[:, 1:]
        sp_yup = torch.stack([sp[:, 0], sp[:, 2], (R - 1) - sp[:, 1]], dim=-1)
        vc = torch.cat([batch_col, sp_yup], dim=-1)
    _log("  tex decode: voxels=", vc.shape[0], "channels=", color.shape[-1], "res=", tex_res)
    return vc[:, 1:], color, tex_res


def _stub_prompt_server():
    """后处理节点会向 ComfyUI 的 PromptServer 推进度；独立脚本里补一个空桩。"""
    try:
        from server import PromptServer

        class _Noop:
            def send_progress_text(self, *a, **k):
                return None

            def send_progress(self, *a, **k):
                return None

            def send_sync(self, *a, **k):
                return None

        if getattr(PromptServer, "instance", None) is None:
            PromptServer.instance = _Noop()
    except Exception:
        pass


def export_textured_glb(out_path, vertices, faces, vox_coords, vox_colors, tex_res,
                        tex_size=1024, open_fn=None, unwrap_resolution=1024):
    """UV 展开 → 烘焙 base_color / metallic / roughness → 带 PBR 材质的 GLB。

    直接调用 ComfyUI 官方后处理节点（节点方法取别名只是局部变量，不涉及任何
    数据库/查询代码）。网格与体素都需是 y_up 坐标系（调用方负责转换）。
    """
    import comfy.model_management as cmm

    # 后处理在 CPU 上算：一是绕开 MPS 的稀疏卷积/解码缺陷（与 VAE 同一类风险），
    # 二是 16GB 统一内存下 MPS 已无余量。
    _orig_get = cmm.get_torch_device
    cmm.get_torch_device = lambda *a, **k: torch.device("cpu")
    try:
        from comfy_api.latest import Types
        from comfy_extras.nodes_mesh_postprocess import (
            UnwrapMesh, BakeTextureFromVoxel, ApplyTextureToMesh)
        from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes

        unwrap_fn = UnwrapMesh.execute
        bake_fn = BakeTextureFromVoxel.execute
        apply_fn = ApplyTextureToMesh.execute

        # 这些 IO.ComfyNode 在官方图里由运行时注入 hidden（unique_id 等）；
        # 独立脚本调用时手动补一个最小上下文，否则 cls.hidden 为 None。
        import types as _types
        for _cls in (UnwrapMesh, BakeTextureFromVoxel, ApplyTextureToMesh):
            if getattr(_cls, "hidden", None) is None:
                _cls.hidden = _types.SimpleNamespace(unique_id="1", prompt=None,
                                                     extra_pnginfo=None)

        verts = torch.as_tensor(vertices, dtype=torch.float32).unsqueeze(0)
        tris = torch.as_tensor(faces, dtype=torch.int64).unsqueeze(0)
        mesh = Types.MESH(vertices=verts, faces=tris)

        _log("  uv unwrap (xatlas, cpu)…")
        _stub_prompt_server()
        mesh_uv = unwrap_fn(mesh, "adaptive", unwrap_resolution, 1, 0.0)[0]
        uvs = mesh_uv.uvs
        uv_shape = tuple(uvs.shape) if uvs is not None else None
        _log("  unwrap: verts=", mesh_uv.vertices.shape[1], "uvs=", uv_shape)

        voxel = Types.VOXEL(vox_coords, vox_colors, tex_res)
        _log("  bake textures", tex_size, "x", tex_size)
        baked = bake_fn(mesh_uv, voxel, tex_size)
        base, metal, rough = baked[0], baked[1], baked[2]
        mesh_tex = apply_fn(mesh_uv, base, metal, rough)[0]
        _log("  apply texture: has_texture=", mesh_tex.texture is not None,
             "has_mr=", mesh_tex.metallic_roughness is not None)

        glb = mesh_item_to_glb_bytes(mesh_tex, 0, {"generator": "trellis2-mac-driver"})
        if glb is None:
            raise RuntimeError("mesh_item_to_glb_bytes 返回空（网格空？）")
        opener = open_fn or (lambda p, m: open(p, m))
        with opener(out_path, "wb") as f:
            f.write(glb)
        _log("  GLB(贴图+材质) 写出:", out_path, len(glb), "bytes")
        return len(glb)
    finally:
        cmm.get_torch_device = _orig_get


def load_texture_vae(ckpt_path, dt, device):
    """加载 TextureVae（txt_dec）。"""
    import comfy.utils
    import comfy.ldm.trellis2.vae as t2v

    sd = comfy.utils.load_torch_file(ckpt_path)
    vae = t2v.TextureVae().to(dtype=dt)
    vae.load_state_dict(sd, strict=True)
    vae.eval().to(device)
    del sd
    return vae
