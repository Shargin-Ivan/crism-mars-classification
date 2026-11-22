"""Verify the portable experiment's immutable workspace (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath


HERE = Path(__file__).resolve().parent
INVENTORY = "input_inventory.json"
BLOCK = 8 * 1024 * 1024


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            checksum.update(chunk)
        return checksum.hexdigest()


def portable_path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PureWindowsPath(relative).drive or relative.startswith("/")
            or any(part in ("", ".", "..") or ":" in part for part in relative.split("/"))):
        raise ValueError(f"Invalid portable path {relative!r}")
    path = root / relative
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError(f"Linked file or directory: {part}")
        if part == root:
            break
    return path


def verify(workspace):
    """Verify immutable inputs offline; generated campaign outputs may coexist."""
    manifest = read(portable_path(workspace, INVENTORY))
    if manifest["schema_version"] != 1 or not manifest["files"]:
        raise ValueError("Empty or unsupported inventory")
    names = set()
    total = 0
    for item in manifest["files"]:
        name = item["path"]
        path = portable_path(workspace, name)
        if (name in names or not path.is_file() or path.stat().st_size != item["size"]
                or digest(path) != item["sha256"]):
            raise ValueError(f"Inventory mismatch: {name}")
        names.add(name)
        total += item["size"]
    if total != manifest["total_bytes"]:
        raise ValueError("Inventory byte count differs")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    action = parser.parse_args().command
    destination = HERE / "workspace"
    result = verify(destination)
    print(f"{len(result['files'])} files, {result['total_bytes']} bytes")
