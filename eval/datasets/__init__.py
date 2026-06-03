from eval.datasets import advbench, harmbench, jailbreakbench

LOADERS = {
    "advbench": advbench.load,
    "harmbench": harmbench.load,
    "jbb": jailbreakbench.load,
}


def load(name: str, limit: int | None = None):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(LOADERS)}")
    items = LOADERS[name]()
    if limit is not None:
        items = items[:limit]
    return items
