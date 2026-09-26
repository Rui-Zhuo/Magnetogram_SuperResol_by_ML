"""Create new deterministic splits; use the provided manifest for the paper."""
import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    files = sorted(x.name for x in a.data.glob('*.npz'))
    if len(files) < 5: p.error('At least five samples are required')
    random.Random(a.seed).shuffle(files)
    n, v = int(.6*len(files)), int(.2*len(files))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as out:
        json.dump(dict(train=files[:n], val=files[n:n+v], test=files[n+v:]), out, indent=2)


if __name__ == '__main__': main()
