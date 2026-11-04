# -*- coding: utf-8 -*-
"""GLB 结构审计：解析 JSON chunk，打印 verts/tris/bbox/generator，粗检网格健康。

用法: python audit_glb.py <file.glb> [more.glb ...]
"""
import json
import struct
import sys


def parse_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, ver, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67, "not GLB"
    off = 12
    chunks = {}
    while off < length:
        clen, ctype = struct.unpack("<II", data[off:off + 8])
        cdata = data[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            chunks["JSON"] = json.loads(cdata.decode("utf-8"))
        elif ctype == 0x004E4942:
            chunks["BIN"] = cdata
        off += 8 + clen + ((4 - clen % 4) % 4)
    return chunks


def audit(path):
    c = parse_glb(path)
    g = c["JSON"]
    print("=" * 64)
    print(path, "%d bytes" % (12 + 8 + len(json.dumps(g)) + 8 + len(c.get("BIN", b""))))
    print("  generator :", g.get("asset", {}).get("generator"))
    meshes = g.get("meshes", [])
    tot_v = tot_t = 0
    bmin = [1e9] * 3
    bmax = [-1e9] * 3
    degenerate = 0
    for m in meshes:
        for prim in m.get("primitives", []):
            a = g["accessors"][prim["attributes"]["POSITION"]]
            tot_v += a["count"]
            if a.get("min"):
                bmin = [min(b, x) for b, x in zip(bmin, a["min"])]
                bmax = [max(b, x) for x, b in zip(bmax, a["max"])]
            if "indices" in prim:
                tot_t += g["accessors"][prim["indices"]]["count"] // 3
    print("  vertices  :", tot_v)
    print("  triangles :", tot_t)
    print("  bbox min  :", [round(x, 4) for x in bmin])
    print("  bbox max  :", [round(x, 4) for x in bmax])
    print("  bbox size :", [round(bmax[i] - bmin[i], 4) for i in range(3)])
    # 简单退化面检查：读 BIN 里的 indices，统计三顶点重复的三角形
    bin_chunk = c.get("BIN")
    if bin_chunk and tot_t:
        for m in meshes:
            for prim in m.get("primitives", []):
                if "indices" not in prim:
                    continue
                ia = g["accessors"][prim["indices"]]
                bv = g["bufferViews"][ia["bufferView"]]
                off = bv.get("byteOffset", 0)
                n = ia["count"]
                idx = struct.unpack_from("<%dI" % n, bin_chunk, off)
                degenerate += sum(1 for i in range(0, n, 3)
                                  if idx[i] == idx[i+1] or idx[i+1] == idx[i+2] or idx[i] == idx[i+2])
    print("  degenerate tris:", degenerate)

    # ---- 贴图 / 材质 ----
    images = g.get("images", [])
    textures = g.get("textures", [])
    materials = g.get("materials", [])
    print("  images    :", len(images))
    print("  textures  :", len(textures))
    print("  materials :", len(materials))
    for mi, mat in enumerate(materials):
        pbr = mat.get("pbrMetallicRoughness", {})
        bct = pbr.get("baseColorTexture")
        mrt = pbr.get("metallicRoughnessTexture")
        print("    mat[%d] name=%r baseColorTex=%s mrTex=%s baseColorFactor=%s "
              "metallic=%s roughness=%s"
              % (mi, mat.get("name"), bct is not None, mrt is not None,
                 pbr.get("baseColorFactor"), pbr.get("metallicFactor"),
                 pbr.get("roughnessFactor")))
    has_uv = any("TEXCOORD_0" in (prim.get("attributes") or {})
                 for m in meshes for prim in m.get("primitives", []))
    print("  TEXCOORD_0:", has_uv)
    # 检查贴图数据量（bufferView 覆盖的字节）
    img_bytes = 0
    for im in images:
        bv_i = im.get("bufferView")
        if bv_i is not None:
            img_bytes += g["bufferViews"][bv_i].get("byteLength", 0)
    print("  image data:", img_bytes, "bytes")
    textured = bool(images) and bool(materials) and has_uv
    print("  textured  :", textured)
    healthy = tot_v > 1000 and tot_t > 1000 and degenerate < tot_t * 0.05
    print("  verdict   :", "HEALTHY" if healthy else "SUSPECT/BROKEN")
    return healthy

if __name__ == "__main__":
    ok = all(audit(p) for p in sys.argv[1:])
    sys.exit(0 if ok else 1)
