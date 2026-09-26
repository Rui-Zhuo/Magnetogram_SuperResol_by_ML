"""Load or reconstruct the existing full-disk results stored in row slabs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_result(case_directory, key='SRFfield', verify=True):
    root = Path(case_directory).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if key not in manifest['arrays']:
        raise ValueError(f'Unknown array {key}; available: {list(manifest["arrays"])}')
    result = np.empty(manifest['shape'], dtype=manifest['arrays'][key]['dtype'])
    next_row = 0
    for part in manifest['parts']:
        file = (root/part['file']).resolve()
        if not file.is_relative_to(root):
            raise ValueError('Part path is outside the case directory')
        if verify and hashlib.sha256(file.read_bytes()).hexdigest() != part['sha256']:
            raise ValueError(f'Checksum mismatch: {file.name}')
        start, stop = part['row_start'], part['row_stop']
        if start != next_row or not start < stop <= result.shape[0]:
            raise ValueError('Missing, overlapping or out-of-order row slabs')
        with np.load(file, allow_pickle=False) as data:
            slab = data[key]
            if slab.shape != (stop-start, result.shape[1]) or slab.dtype != result.dtype:
                raise ValueError(f'Unexpected array schema in {file.name}')
            result[start:stop] = slab
        next_row = stop
    if next_row != result.shape[0]:
        raise ValueError('Incomplete full-disk array')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    arrays = {key: load_result(a.case, key) for key in ['SRfield', 'SRFfield']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **arrays)
    print(f'Saved existing full-disk arrays to {a.output}')


if __name__ == '__main__':
    main()
