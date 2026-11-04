# -*- coding: utf-8 -*-
"""在 applemini 本机 ComfyUI(127.0.0.1:8188) 上产 red_brick 裁切图。

与 GPU 基准同配方：RemoveBackground(birefnet) -> ImageCropToMask(1024, pad_factor=1.0)。
产物: ~/ComfyUI/output/drv/crop_red_brick_00001_.png
仅访问本机回环地址。
"""
import ipaddress
import json
import os
import socket
import time
import urllib.request

HOST = "127.0.0.1"
PORT = 8188


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect, urllib.request.ProxyHandler({}))


def _check_url(url):
    from urllib.parse import urlsplit
    sp = urlsplit(url)
    if sp.scheme != "http" or sp.hostname != HOST or sp.port != PORT:
        raise SystemExit("URL 不在白名单内: %r" % url)
    for _f, _s, _p, _cn, sa in socket.getaddrinfo(HOST, PORT, proto=socket.IPPROTO_TCP):
        ip = ipaddress.ip_address(sa[0])
        if not ip.is_loopback:
            raise SystemExit("解析出非环回地址: %s" % ip)
    return url


def get_json(path):
    with _OPENER.open(_check_url("http://%s:%d%s" % (HOST, PORT, path)), timeout=30) as r:
        return json.loads(r.read().decode())


def post_json(path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(_check_url("http://%s:%d%s" % (HOST, PORT, path)), data=data,
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    info = get_json("/object_info")
    for cls in ("RemoveBackground", "ImageCropToMask", "LoadBackgroundRemovalModel", "SaveImage"):
        if cls not in info:
            raise SystemExit("ComfyUI 缺少节点: " + cls)
    # 校验 ImageCropToMask 有 grow_mask/background（0.43 schema）
    req = info["ImageCropToMask"]["input"]["required"]
    optional = info["ImageCropToMask"]["input"].get("optional", {})
    wf = {"prompt": {
        "10": {"inputs": {"image": "red_brick.png"}, "class_type": "LoadImage"},
        "20": {"inputs": {"bg_removal_name": "birefnet.safetensors"},
               "class_type": "LoadBackgroundRemovalModel"},
        "30": {"inputs": {"bg_removal_model": ["20", 0], "image": ["10", 0]},
               "class_type": "RemoveBackground"},
        "40": {"inputs": {"images": ["10", 0], "masks": ["30", 0], "width": 1024, "height": 1024,
                          "pad_factor": 1.0},
               "class_type": "ImageCropToMask"},
        "50": {"inputs": {"images": ["40", 0], "filename_prefix": "drv/crop_red_brick"},
               "class_type": "SaveImage"},
    }}
    if "grow_mask" in req:
        wf["prompt"]["40"]["inputs"]["grow_mask"] = 0
    if "background" in req or "background" in optional:
        wf["prompt"]["40"]["inputs"]["background"] = "#000000"

    resp = post_json("/prompt", {"prompt": wf["prompt"], "client_id": "crop-mac"})
    pid = resp["prompt_id"]
    print("submitted", pid, flush=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        time.sleep(4)
        try:
            h = get_json("/history/%s" % pid)
        except Exception:
            continue
        if pid in h:
            e = h[pid]
            st = e.get("status", {})
            if st.get("completed"):
                print("DONE", json.dumps(e.get("outputs", {}).get("50", {}).get("images", [])))
                return
            if st.get("status_str") == "error":
                raise SystemExit("ERROR " + json.dumps(e)[:1500])
    raise SystemExit("TIMEOUT")


if __name__ == "__main__":
    main()
