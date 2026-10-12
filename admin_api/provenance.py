import hashlib
import subprocess

from admin_api.store import ROOT


def code_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_hash():
    digest = hashlib.sha256()
    for directory in ("admin_api", "benchmarks", "app"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
