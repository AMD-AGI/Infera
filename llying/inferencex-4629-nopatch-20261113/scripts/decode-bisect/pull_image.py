#!/usr/bin/env python3
"""Build an enroot-style rootfs for a Docker Hub image without dockerd or root.

Downloads every layer blob anonymously (in parallel), applies them in order with OCI whiteouts,
keeps owner write permission on everything (read-only directories in later layers otherwise
break extraction, the failure rootless `enroot import` hits on lmsysorg images), and writes the
files enroot adds on import (/etc/environment from the image Env, /etc/rc, /etc/fstab,
/etc/rc.local). Squash it afterwards with mksquashfs.

usage: pull_image.py <repo> <digest> <rootfs dir> <blob dir>
"""
import concurrent.futures as cf
import json
import os
import shutil
import stat
import sys
import tarfile
import time
import urllib.request

REG = "https://registry-1.docker.io"


def token(repo):
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    return json.load(urllib.request.urlopen(url))["token"]


def get(repo, path, accept=None, out=None):
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{REG}/v2/{repo}/{path}", headers={"Authorization": f"Bearer {token(repo)}"})
            if accept:
                req.add_header("Accept", accept)
            with urllib.request.urlopen(req, timeout=600) as r:
                if out is None:
                    return r.read()
                tmp = out + ".part"
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, 16 << 20)
                os.replace(tmp, out)
                return None
        except Exception as e:
            print(f"  retry {attempt + 1} {path[:40]}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed {path}")


def rm(p):
    if os.path.islink(p) or (os.path.exists(p) and not os.path.isdir(p)):
        os.unlink(p)
    elif os.path.isdir(p):
        for root, dirs, _ in os.walk(p):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), 0o755)
                except OSError:
                    pass
        shutil.rmtree(p)


def apply_layer(blob, rootfs):
    seen = set()
    with tarfile.open(blob, "r|*") as tf:
        for m in tf:
            name = os.path.normpath(m.name.lstrip("/"))
            if name in (".", ""):
                continue
            base, leaf = os.path.split(name)
            dest_dir = os.path.join(rootfs, base)
            if leaf == ".wh..wh..opq":
                if os.path.isdir(dest_dir):
                    for child in os.listdir(dest_dir):
                        if os.path.join(base, child) not in seen:
                            rm(os.path.join(dest_dir, child))
                continue
            if leaf.startswith(".wh."):
                rm(os.path.join(dest_dir, leaf[4:]))
                continue
            seen.add(name)
            dest = os.path.join(rootfs, name)
            parent = os.path.dirname(dest)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            if os.path.isdir(parent):
                os.chmod(parent, os.stat(parent).st_mode | stat.S_IRWXU)
            if m.isdir():
                if os.path.lexists(dest) and not os.path.isdir(dest):
                    os.unlink(dest)
                os.makedirs(dest, exist_ok=True)
                os.chmod(dest, (m.mode & 0o7777) | stat.S_IRWXU)
                continue
            if os.path.lexists(dest):
                rm(dest)
            if m.ischr() or m.isblk() or m.isfifo():
                continue
            m.name = name
            m.mode = (m.mode & 0o7777) | stat.S_IRUSR | stat.S_IWUSR
            if m.islnk():
                m.linkname = os.path.normpath(m.linkname.lstrip("/"))
            try:
                tf.extract(m, rootfs, set_attrs=True, numeric_owner=True, filter="fully_trusted")
            except TypeError:
                tf.extract(m, rootfs, set_attrs=True, numeric_owner=True)
            except (OSError, tarfile.ExtractError) as e:
                print(f"  skip {name}: {e}", flush=True)


def main():
    repo, digest, rootfs, blobs = sys.argv[1:5]
    os.makedirs(rootfs, exist_ok=True)
    os.makedirs(blobs, exist_ok=True)
    man = json.loads(get(repo, f"manifests/{digest}",
                         "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"))
    cfg = json.loads(get(repo, f"blobs/{man['config']['digest']}"))
    layers = man["layers"]
    print(f"{len(layers)} layers, {sum(l['size'] for l in layers) / 1e9:.1f} GB", flush=True)
    t0 = time.time()

    def fetch(l):
        out = os.path.join(blobs, l["digest"].split(":")[1])
        if not (os.path.exists(out) and os.path.getsize(out) == l["size"]):
            get(repo, f"blobs/{l['digest']}", out=out)
        return out

    with cf.ThreadPoolExecutor(12) as ex:
        paths = list(ex.map(fetch, layers))
    print(f"downloaded in {time.time() - t0:.0f}s", flush=True)
    for i, p in enumerate(paths):
        t = time.time()
        apply_layer(p, rootfs)
        print(f"layer {i + 1}/{len(paths)} applied ({os.path.getsize(p) / 1e6:.0f} MB, {time.time() - t:.0f}s)", flush=True)

    c = cfg.get("config", {})
    etc = os.path.join(rootfs, "etc")
    os.makedirs(etc, exist_ok=True)
    with open(os.path.join(etc, "environment"), "w") as f:
        for e in c.get("Env") or []:
            f.write(e + "\n")
    wd = c.get("WorkingDir") or "/"
    ep = " ".join(f"'{x}'" for x in (c.get("Entrypoint") or []))
    cmd = " ".join(f"'{x}'" for x in (c.get("Cmd") or ["/bin/bash"]))
    labels = c.get("Labels") or {}
    with open(os.path.join(etc, "rc"), "w") as f:
        for k, v in labels.items():
            f.write(f"# {k} {v}\n")
        f.write(f'\nmkdir -p "{wd}" 2> /dev/null\ncd "{wd}" && unset OLDPWD || exit 1\n\n'
                "if [ -s /etc/rc.local ]; then\n    . /etc/rc.local\nfi\n\n"
                f'if [ $# -gt 0 ]; then\n    exec {ep} "$@"\nelse\n    exec {ep} {cmd}\nfi\n')
    for n in ("fstab", "rc.local"):
        open(os.path.join(etc, n), "a").close()
    print(f"rootfs ready in {time.time() - t0:.0f}s; workdir {wd}", flush=True)


if __name__ == "__main__":
    main()
