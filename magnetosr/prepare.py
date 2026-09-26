"""Refine coarse alignment and create the paired L72 NPZ format."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
from .metrics import bicubic
from .registration import local_subpixel_registration


def prepare_pair(path, output, window=32, step=1):
    with np.load(path, allow_pickle=False) as d:
        hmi = np.array(d['HMIfield_crop'], dtype=float)
        sp = np.array(d['SPfield_coalign'], dtype=float)
        radius = np.array(d['Txy_crop'], dtype=float)
    if hmi.shape != radius.shape or min(hmi.shape) < 200:
        raise ValueError('Require matching HMI/Txy fields with both dimensions >= 200')
    if not all(np.isfinite(x).all() for x in [hmi, sp, radius]):
        raise ValueError('Non-finite field; excluded rather than silently filled')
    # Preserve the historical sequence: SP -> HMI grid -> local registration
    # -> centre crop -> anisotropic bicubic resampling.
    sp_on_hmi = bicubic(sp, hmi.shape)
    aligned, dx, dy = local_subpixel_registration(hmi, sp_on_hmi, window=window, step=step)
    y, x = (np.array(hmi.shape)-200)//2
    crop = np.s_[y:y+200, x:x+200]
    target = bicubic(aligned[crop], (312, 336))
    np.savez_compressed(output, HMIfield=hmi[crop], SPfield=target, Txy=radius[crop])
    return dict(sample=path.name, dx_median=float(np.median(dx)), dy_median=float(np.median(dy)),
                displacement_p95_pixels=float(np.percentile(np.hypot(dx, dy), 95)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--split', type=Path, help='Restrict to IDs in an existing split; never overwrite it')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    selected = None
    if a.split:
        selected = {f for names in json.loads(a.split.read_text()).values() for f in names}
    records, failures = [], []
    files = sorted(a.input.glob('*.npz'))
    if not files: p.error('No coarse NPZ inputs found')
    for path in files:
        if selected is not None and path.name not in selected: continue
        try:
            records.append(prepare_pair(path, a.output/path.name, a.window, a.step))
            print(path.name, flush=True)
        except Exception as error:
            failures.append(dict(sample=path.name, reason=str(error)))
    if selected is not None:
        seen = {r['sample'] for r in records} | {r['sample'] for r in failures}
        failures.extend(dict(sample=x, reason='Coarse input missing') for x in sorted(selected-seen))
    (a.output/'preparation_report.json').write_text(json.dumps(dict(processed=records, failed=failures), indent=2))
    if failures: raise SystemExit(f'{len(failures)} samples failed; see preparation_report.json')


if __name__ == '__main__': main()
