from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent


def parse_requirements(filename: str) -> tuple[list[str], list[str], list[str]]:
    path = ROOT / filename
    if not path.exists():
        return [], [], []
    base: list[str] = []
    train: list[str] = []
    test: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "; extra ==" in line:
            req, marker = line.split(";", 1)
            req = req.strip()
            marker = marker.strip()
            if "extra == \"train\"" in marker or "extra == 'train'" in marker:
                train.append(req)
                continue
            if "extra == \"test\"" in marker or "extra == 'test'" in marker:
                test.append(req)
                continue
        base.append(line)
    return base, train, test


def unique(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


base_requires, train_requires, test_requires = parse_requirements("requirements.txt")

setup(
    name="asft-dev",
    version="0.0.0",
    description="ASFT development environment",
    packages=[],
    install_requires=base_requires,
    extras_require={
        "train": unique(train_requires),
        "test": unique(train_requires + test_requires),
    },
)
