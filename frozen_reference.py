"""Load the pinned analysis subset of the completed original H100 campaign."""
from package_workspace import digest, read


def verified_steps(root):
    path = root / "campaign.json"
    if digest(path) != "39470a9953d83f8d3a6b3683e16d3ae2fae31f5a918d74f873e0385ec92fbdc3":
        raise ValueError("Original 25+25 reference index differs")
    steps = read(path)["steps"]
    for record in steps.values():
        folder = root / record["analysis"]
        for name, sha in record["pins"].items():
            if digest(folder / name) != sha:
                raise ValueError(f"Original 25+25 reference changed: {folder.name}/{name}")
    return steps
