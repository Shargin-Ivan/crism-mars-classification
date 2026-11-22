"""Restore upstream LWNet files missing from a source-only clone."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

from package_workspace import digest, read


WORK = Path(__file__).resolve().parent / "workspace"
BASE = "https://raw.githubusercontent.com/hkzhang-git/LWNet/ca8361cd4a1c6418e3366fcc2ebc91fb72ac8c78/"


def main():
    inventory = {r["path"]: r["sha256"] for r in read(WORK / "input_inventory.json")["files"]}
    for path, names in (
        ("training/models/models_lw_3D.py", ["LWNet/training/models/models_lw_3D.py", "finetuning/models/vendor/lwnet.py"]),
        ("training/transfer_model/Salinas_model.pkl", ["LWNet/training/transfer_model/Salinas_model.pkl"]),
    ):
        data = None
        for name in names:
            target = WORK / name
            if target.exists() and digest(target) == inventory[name]:
                data = target.read_bytes()
                continue
            if data is None:
                with urlopen(BASE + path, timeout=120) as response:
                    data = response.read()
                if path.endswith(".py"):
                    data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if hashlib.sha256(data).hexdigest() != inventory[name]:
                raise ValueError(f"Upstream SHA-256 mismatch: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print("LWNet source and Salinas weights verified")


if __name__ == "__main__":
    main()
