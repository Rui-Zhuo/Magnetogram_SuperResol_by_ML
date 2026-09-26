"""One evaluator for every model, with explicit membership and angle bins."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .metrics import evaluate, bicubic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--metadata', type=Path, default=Path('data/test_metadata.csv'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    metadata = {x['sample']: x for x in csv.DictReader(a.metadata.open(encoding='utf-8-sig'))}
    rows = []
    for file in sorted(a.predictions.glob('*.npz')):
        if file.name not in metadata:
            raise ValueError(f'{file.name}: missing test-set membership metadata')
        with np.load(a.data/file.name, allow_pickle=False) as d:
            gt, hmi = d['SPfield'], d['HMIfield']
        with np.load(file, allow_pickle=False) as d:
            pred = d['pred'].reshape(gt.shape)
        theta = float(metadata[file.name]['theta_deg'])
        radius = np.sin(np.deg2rad(theta))
        if not 0 <= theta <= 90:
            raise ValueError(f'Invalid heliocentric angle for {file.name}')
        angle_bin = '0-0.5' if radius < .5 else '0.5-0.8' if radius < .8 else '0.8-1.0'
        for name, field in [('prediction', pred), ('bicubic', bicubic(hmi, gt.shape))]:
            rows.append(dict(sample=file.name, method=name, region=metadata[file.name]['region_type'],
                             radius_bin=angle_bin, **evaluate(field, gt)))
    if not rows:
        p.error('No prediction NPZ files found')
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'per_sample.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summaries = []
    metrics = list(evaluate(np.eye(8), np.eye(8)))
    for method in ['prediction', 'bicubic']:
        for dimension, labels in [('all', ['all']), ('region', ['AR', 'non-AR']),
                                  ('radius_bin', ['0-0.5', '0.5-0.8', '0.8-1.0'])]:
            for label in labels:
                subset = [r for r in rows if r['method']==method and (dimension=='all' or r[dimension]==label)]
                if not subset: continue
                item = dict(method=method, grouping=dimension, group=label, n=len(subset))
                for metric in metrics:
                    values = np.array([r[metric] for r in subset]); finite = values[np.isfinite(values)]
                    item[metric] = dict(n_finite=len(finite), mean=float(finite.mean()) if len(finite) else None,
                                        std_population=float(finite.std()) if len(finite) else None)
                for metric in ['net_flux_ratio', 'unsigned_flux_ratio']:
                    mean = item[metric]['mean']
                    item[metric.replace('ratio', 'score')] = None if mean is None else 1-abs(mean-1)
                summaries.append(item)
    (a.output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    print(f'Evaluated {len(rows)//2} test samples; output: {a.output}')


if __name__ == '__main__': main()
