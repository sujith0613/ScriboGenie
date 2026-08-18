#!/usr/bin/env python3
"""podmansetup.py — one-shot Podman setup for the ScriboGenie Raspberry Pi.

Pre-flights the Pi, installs Podman + rootless dependencies via apt, configures
the target user for rootless operation, and verifies the result. Idempotent —
safe to re-run.

Python 3 stdlib only (no pip dependencies). Must run as root (uses sudo).

Usage:
    sudo python3 deploy/podmansetup.py            # full install + configure + verify
    sudo python3 deploy/podmansetup.py --check    # pre-flight only, no changes
    sudo python3 deploy/podmansetup.py --user pi  # rootless user override
    sudo python3 deploy/podmansetup.py --verbose  # print subcommand output
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PODMAN = (4, 3, 0)
MIN_DISK_MB = 5 * 1024
MIN_RAM_MB = 2 * 1024
SUPPORTED_ARCHES = {"aarch64", "arm64"}
ACCEPTED_IDS = {"debian", "raspbian", "raspios", "armbian"}
SUBUID_RANGE = (100000, 165535)

VERBOSE = False


def log(msg: str = "") -> None:
    print(msg)


def warn(msg: str) -> None:
    print(f"  WARN: {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    sys.exit(1)


def run(cmd, check: bool = True, text: bool = True, input=None, capture: bool = False):
    """Run a command; optionally capture stdout, optionally ignore failure."""
    if VERBOSE:
        log(f"  $ {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, text=text, input=input, capture_output=capture,
            stdout=subprocess.DEVNULL if not capture and not VERBOSE else None,
            stderr=subprocess.DEVNULL if not capture and not VERBOSE else None,
        )
    except FileNotFoundError:
        if check:
            fail(f"command not found: {cmd[0]}")
        return None
    if check and proc.returncode != 0:
        if capture:
            log(proc.stderr.strip() if text else "")
        fail(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def detect_os():
    info = {"id": None, "codename": None, "version": None, "name": None}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip().strip('"')
            if k in ("ID", "VERSION_CODENAME", "VERSION_ID", "PRETTY_NAME"):
                info[{"ID": "id", "VERSION_CODENAME": "codename",
                      "VERSION_ID": "version", "PRETTY_NAME": "name"}[k]] = v
    except OSError:
        pass
    return info


def cgroups_v2() -> bool:
    return Path("/sys/fs/cgroup/cgroup.controllers").is_file()


def disk_free_mb(path: str) -> int:
    s = os.statvfs(path)
    return (s.f_bavail * s.f_frsize) // (1024 * 1024)


def mem_total_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def parse_podman_version(text: str):
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


def user_subids(user: str, gid: bool) -> bool:
    fname = "/etc/subgid" if gid else "/etc/subuid"
    try:
        for line in Path(fname).read_text().splitlines():
            if line.startswith(f"{user}:"):
                lo, hi = line.split(":")[1], line.split(":")[2]
                if int(lo) <= SUBUID_RANGE[0] and int(lo) + int(hi) >= SUBUID_RANGE[1]:
                    return True
        return False
    except OSError:
        return False


def apt_install(packages, apt_ok):
    run(["apt-get", "update"], check=apt_ok)
    cmd = ["apt-get", "install", "-y", "--no-install-recommends"] + packages
    run(cmd, check=apt_ok)


def main():
    global VERBOSE
    ap = argparse.ArgumentParser(description="ScriboGenie Pi Podman setup")
    ap.add_argument("--check", action="store_true",
                    help="pre-flight only; make no changes")
    ap.add_argument("--yes", action="store_true",
                    help="non-interactive (apt never prompts anyway)")
    ap.add_argument("--verbose", action="store_true", help="print subcommand output")
    ap.add_argument("--user", default=None, help="rootless target user (default: SUDO_USER)")
    args = ap.parse_args()
    VERBOSE = args.verbose

    if os.geteuid() != 0:
        fail("must run as root — use: sudo python3 deploy/podmansetup.py")

    if args.user:
        target_user = args.user
    else:
        target_user = os.environ.get("SUDO_USER") or os.environ.get("USER")
    if not target_user:
        fail("could not determine target user — pass --user")

    log("=" * 68)
    log("ScriboGenie Pi — Podman setup")
    log("=" * 68)
    log(f"Target user: {target_user}")
    log()

    # ---- 1. Detect ---------------------------------------------------------
    os_info = detect_os()
    arch = platform.machine()
    kernel = platform.release()
    cg2 = cgroups_v2()
    disk_mb = disk_free_mb("/var/lib/containers" if os.path.isdir("/var/lib/containers") else "/")
    mem_mb = mem_total_mb()

    log("[detect]")
    log(f"  distro   : {os_info['name'] or 'unknown'} (id={os_info['id']}, codename={os_info['codename']})")
    log(f"  arch     : {arch}")
    log(f"  kernel   : {kernel}")
    log(f"  cgroups  : {'v2' if cg2 else 'v1/unified'}")
    log(f"  disk free: {disk_mb} MB")
    log(f"  RAM      : {mem_mb} MB")
    log()

    # ---- 2. Compatibility gates -------------------------------------------
    log("[checks]")
    ok = True

    if arch in SUPPORTED_ARCHES:
        log("  PASS  architecture is aarch64/arm64")
    else:
        ok = False
        warn(f"arch {arch} cannot run the arm64 container image — "
             "install Raspberry Pi OS 64-bit")

    if os_info["id"] in ACCEPTED_IDS or (os_info["id"] and "debian" in (os_info["id"] or "")):
        log(f"  PASS  distro {os_info['id']} is Debian-based")
    else:
        ok = False
        warn(f"unsupported distro id: {os_info['id']}")

    if cg2:
        log("  PASS  cgroups v2 present (required by Podman 5)")
    else:
        ok = False
        warn("cgroups v2 not active — add 'systemd.unified_cgroup_hierarchy=1' "
             "to /boot/firmware/cmdline.txt (Trixie) and reboot")

    if disk_mb >= MIN_DISK_MB:
        log(f"  PASS  disk free {disk_mb} MB >= {MIN_DISK_MB} MB")
    else:
        ok = False
        warn(f"only {disk_mb} MB free — need >= {MIN_DISK_MB} MB for image + layers")

    if mem_mb >= MIN_RAM_MB:
        log(f"  PASS  RAM {mem_mb} MB >= {MIN_RAM_MB} MB")
    else:
        warn(f"RAM {mem_mb} MB is low (recommended >= {MIN_RAM_MB} MB)")

    # podman already installed? version check.
    podman_bin = shutil.which("podman")
    installed_ver = None
    if podman_bin:
        out = run(["podman", "--version"], check=False, capture=True)
        if out and out.stdout:
            installed_ver = parse_podman_version(out.stdout)
    if installed_ver:
        if installed_ver >= MIN_PODMAN:
            log(f"  PASS  podman {'.'.join(map(str, installed_ver))} >= "
                f"{'.'.join(map(str, MIN_PODMAN))}")
        else:
            ok = False
            warn(f"podman {'.'.join(map(str, installed_ver))} is too old — "
                 "upgrade the distro or apt")
    else:
        log("  INFO  podman not installed — will install via apt")

    if not ok:
        log()
        fail("pre-flight failed — fix the items above and re-run")

    # ---- 3. check mode stops here -----------------------------------------
    if args.check:
        log()
        log("Pre-flight complete: all checks passed. System ready for podman install.")
        log("Run without --check to install and configure.")
        sys.exit(0)

    # ---- 4. Install ---------------------------------------------------------
    # Skip the apt install entirely when a compatible podman is already present
    # (avoids a silent multi-minute `apt-get update`+install on every re-run).
    if installed_ver and installed_ver >= MIN_PODMAN:
        log("[install]")
        log(f"  podman {'.'.join(map(str, installed_ver))} already installed — skipping apt install")
    else:
        log()
        log("[install]")
        packages = ["podman", "uidmap", "slirp4netns", "fuse-overlayfs"]
        # Debian 13 Trixie / Raspberry Pi OS ships Podman 5 (pasta networking).
        if os_info["codename"] in ("trixie", "forky", "kingfisher", None):
            packages.append("passt")
        apt_install(packages, apt_ok=True)
        log(f"  installed: {' '.join(packages)}")

    # ---- 5. Rootless config ------------------------------------------------
    log()
    log("[rootless]")
    if not user_subids(target_user, gid=False):
        run(["usermod", "--add-subuids",
             f"{SUBUID_RANGE[0]}-{SUBUID_RANGE[1]}", target_user])
        log(f"  added subuid {SUBUID_RANGE[0]}-{SUBUID_RANGE[1]} for {target_user}")
    else:
        log(f"  subuid already configured for {target_user}")
    if not user_subids(target_user, gid=True):
        run(["usermod", "--add-subgids",
             f"{SUBUID_RANGE[0]}-{SUBUID_RANGE[1]}", target_user])
        log(f"  added subgid {SUBUID_RANGE[0]}-{SUBUID_RANGE[1]} for {target_user}")
    else:
        log(f"  subgid already configured for {target_user}")

    storage_conf = Path(f"/home/{target_user}/.config/containers/storage.conf")
    storage_conf.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "[storage]\n"
        "driver = \"overlay\"\n"
        "\n"
        "[storage.options.overlay]\n"
        "mount_program = \"/usr/bin/fuse-overlayfs\"\n"
    )
    if storage_conf.exists() and storage_conf.read_text().strip() == content.strip():
        log("  storage.conf already correct")
    else:
        storage_conf.write_text(content)
        run(["chown", "-R", target_user, str(storage_conf.parent)])
        log(f"  wrote {storage_conf} (overlay + fuse-overlayfs)")

    # ---- 6. Verify ----------------------------------------------------------
    log()
    log("[verify]")
    out = run(["podman", "--version"], capture=True)
    log(f"  {out.stdout.strip()}")

    info_out = run(["podman", "info",
                    "--format", "{{.Host.Arch}} {{.Host.CgroupsVersion}} "
                                "{{.Store.GraphDriverName}}"],
                   capture=True, check=False)
    if info_out and info_out.stdout:
        fields = info_out.stdout.strip().split()
        arch_v, cg_v = fields[0], fields[1]
        gd = fields[2] if len(fields) > 2 else "?"
        log(f"  arch={arch_v} cgroups={cg_v} graphdriver={gd}")
        if arch_v == "arm64" and cg_v == "v2":
            log("  PASS  podman is arm64 on cgroups v2")
        else:
            warn(f"unexpected podman info: {info_out.stdout.strip()}")
    else:
        warn("could not run 'podman info' as root — "
             "rootless behavior must be checked as the user")

    log()
    log("=" * 68)
    log("Podman setup complete. Next steps on the Pi:")
    log(f"  1. log in as {target_user} and verify rootless:")
    log(f"     su - {target_user} && podman info --format '{{{{.Host.Rootless}}}}'")
    log("  2. load the container image:")
    log("     podman load -i scribogenie.tar  (as the desktop user)")
    log("  3. run the app (host networking, X11, audio, Wacom):")
    log("     podman run --rm --network host \\")
    log("       -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix \\")
    log("       -v /dev/snd:/dev/snd \\")
    log("       -v /dev/input:/dev/input \\")
    log("       -v $PWD/logs:/app/logs -v $PWD/data:/app/data \\")
    log("       scribogenie")
    log("=" * 68)
    sys.exit(0)


if __name__ == "__main__":
    main()
