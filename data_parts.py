"""Pack or restore external data as deduplicated parts of at most 512 MiB."""
import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from package_workspace import BLOCK, digest, portable_path, read


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "workspace"
PART_BYTES = 512 * 1024**2


def transfer(action, folder):
    names = set(read(ROOT / "large_inputs.json"))
    groups = defaultdict(list)
    for row in read(WORK / "input_inventory.json")["files"]:
        if row["path"] in names:
            groups[row["sha256"]].append((portable_path(WORK, row["path"]), row["size"]))
    if sum(map(len, groups.values())) != len(names):
        raise ValueError("Unknown external input")
    if action == "pack":
        folder.mkdir(parents=True, exist_ok=True)
    for checksum, rows in groups.items():
        path, size = rows[0]
        parts = [(folder / f"{checksum}.{i:03}", min(PART_BYTES, size - offset))
                 for i, offset in enumerate(range(0, size, PART_BYTES))]
        if action == "pack":
            if digest(path) != checksum or path.stat().st_size != size:
                raise ValueError(f"Source changed: {path}")
            with path.open("rb") as source:
                for part, remaining in parts:
                    with part.open("wb") as target:
                        while remaining:
                            block = source.read(min(BLOCK, remaining))
                            if not block:
                                raise IOError(f"Short read: {path}")
                            target.write(block)
                            remaining -= len(block)
            continue
        existing = next((file for file, _ in rows if file.is_file() and digest(file) == checksum), None)
        if existing is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
                temporary = Path(target.name)
            try:
                hasher = hashlib.sha256()
                with temporary.open("wb") as target:
                    for part, length in parts:
                        if part.stat().st_size != length:
                            raise ValueError(f"Incorrect part length: {part}")
                        with part.open("rb") as source:
                            for block in iter(lambda: source.read(BLOCK), b""):
                                hasher.update(block)
                                target.write(block)
                if hasher.hexdigest() != checksum:
                    raise ValueError(f"Data checksum mismatch: {path}")
                os.replace(temporary, path)
                existing = path
            finally:
                temporary.unlink(missing_ok=True)
        for file, _ in rows:
            if file == existing or (file.is_file() and digest(file) == checksum):
                continue
            if file.exists():
                raise ValueError(f"Conflicting existing duplicate: {file}")
            file.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(existing, file)
            except OSError:
                shutil.copyfile(existing, file)
    print(f"{action}: {len(groups)} unique inputs, {len(names)} paths")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "restore"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    transfer(args.action, args.directory)
